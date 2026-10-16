"""Encoders that read a tree-packed sequence (``packing.py``).

Both take token ids, rotary position ids and a boolean attention mask of shape (batch, length, length), and return
one vector per token:

* ``TinyEncoder``, a small pre-norm transformer with rotary attention, trained from scratch in the CPU toy;
* ``TreeMaskedModernBert``, which runs a Hugging Face ``ModernBertModel`` (ModernBERT-large in decide-400m) under
  the same mask. ModernBERT alternates global layers with local sliding-window layers; under tree packing the window
  is measured in position ids, so a local layer sees the same neighbourhood a flat sequence would give it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .packing import PAD, STATE


def tree_mask(qgroup: torch.Tensor, ogroup: torch.Tensor) -> torch.Tensor:
    """(batch, length) group labels to the (batch, length, length) boolean attention tree. True means attend."""
    qi, qj = qgroup[:, :, None], qgroup[:, None, :]
    oi, oj = ogroup[:, :, None], ogroup[:, None, :]
    same_question = (qj == qi) & (qi != STATE) & (qi != PAD)
    to_state = qj == STATE
    tree = to_state | (same_question & ((oj == STATE) | (oj == oi)))
    pad_rows = (qi == PAD) & ((qj == STATE) | (qj == PAD))
    return tree | pad_rows


def rotary(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Applies rotary position embedding to (batch, heads, length, dim) by position ids of shape (batch, length)."""
    dim = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = positions[:, None, :, None].float() * inv  # (b, 1, l, dim/2)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.to(x.dtype)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, positions), rotary(k, positions)
        p = self.drop.p if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], dropout_p=p)
        h = h + self.drop(self.out(a.transpose(1, 2).reshape(b, n, d)))
        return h + self.drop(self.mlp(self.norm2(h)))


class TinyEncoder(nn.Module):
    """A small rotary transformer for the CPU toy. ``hidden_size`` matches the Hugging Face attribute name."""

    def __init__(self, vocab_size: int, d: int = 128, layers: int = 2, heads: int = 4, ff: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = d
        self.embed = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(d))
        self.blocks = nn.ModuleList(Block(d, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        for block in self.blocks:
            h = block(h, positions, mask)
        return self.norm(h)


class TreeMaskedModernBert(nn.Module):
    """Runs a Hugging Face ``ModernBertModel`` under a tree mask with tree position ids.

    The model's own forward builds its masks from a 2-D padding mask; this wrapper builds the masks itself (the
    global one from the tree, the local one from the tree and the distance in position ids) and calls the same
    embedding, layers and final norm. It needs ``attn_implementation`` "sdpa" or "eager": flash attention does not
    take an arbitrary mask. Under "sdpa" the masks are boolean, which ``scaled_dot_product_attention`` takes in any
    precision; "eager" gets additive masks. ``checkpointing`` recomputes each layer in the backward pass instead of
    storing its activations, which training on a small GPU needs.
    """

    def __init__(self, model, checkpointing: bool = False):
        super().__init__()
        impl = getattr(model.config, "_attn_implementation", "sdpa")
        if impl not in ("sdpa", "eager"):
            raise ValueError(f"tree masks need sdpa or eager attention, not {impl!r}")
        self.model = model
        self.hidden_size = model.config.hidden_size
        self.checkpointing = checkpointing
        self.additive = impl == "eager"

    @classmethod
    def from_pretrained(cls, name: str = "answerdotai/ModernBERT-large", checkpointing: bool = False,
                        **kwargs) -> "TreeMaskedModernBert":
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(name, attn_implementation="sdpa", **kwargs), checkpointing)

    def resize(self, vocab_size: int) -> None:
        """Makes room for the marker tokens a tokenizer adds ([Q], [OPT] and the three kinds)."""
        self.model.resize_token_embeddings(vocab_size)

    def masks(self, positions: torch.Tensor, mask: torch.Tensor):
        m = self.model
        window = (positions[:, :, None] - positions[:, None, :]).abs() <= m.config.local_attention // 2
        global_mask, local_mask = mask[:, None], (mask & window)[:, None]
        if not self.additive:
            return global_mask, local_mask
        dtype = next(m.parameters()).dtype
        neg = torch.finfo(dtype).min
        zeros = torch.zeros(global_mask.shape, dtype=dtype, device=mask.device)
        return zeros.masked_fill(~global_mask, neg), zeros.masked_fill(~local_mask, neg)

    def forward(self, ids: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = self.model
        global_mask, local_mask = self.masks(positions, mask)
        h = m.embeddings(input_ids=ids)
        for layer in m.layers:
            if self.checkpointing and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    lambda x, layer=layer: layer(x, attention_mask=global_mask, sliding_window_mask=local_mask,
                                                 position_ids=positions)[0], h, use_reentrant=False)
            else:
                h = layer(h, attention_mask=global_mask, sliding_window_mask=local_mask, position_ids=positions)[0]
        return m.final_norm(h)
