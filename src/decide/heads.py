"""The three answer heads. Each reads marker vectors from the encoder (``packing.py``) and returns raw logits;
``calibrate.py`` fits a temperature per kind, and ``model.py`` turns logits into probabilities.

* ``ChoiceHead``. Each option's [OPT] vector has seen the state, its question and its own text, not the other
  options. A set-attention layer over {none, option 1, ..., option K} lets options be compared, and lets a *none
  fits* slot, seeded from the question vector, look at all of them. The layer has no positions, so the head is
  permutation-equivariant in the options. A plain softmax over the given options must spend all its mass on them
  even when none fits; the none slot gives that mass somewhere to go.
* ``YesNoHead``. A Bernoulli logit from the question's [Q] vector.
* ``ScoreHead``. A cumulative-link (proportional-odds) head over the levels, given low to high: P(Y <= j) =
  sigmoid(theta_j - eta). The latent position eta comes from the [Q] vector. The cutpoints come from the levels'
  own vectors, so a scale is read from its labels: theta_1 from the first two levels, and each later cutpoint adds
  a softplus gap, which keeps them increasing whatever the weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


#: The hidden width of the heads' small MLPs: the encoder's width up to 256, so the heads stay small beside it.
HEAD_WIDTH = 256


def mlp(d: int, out: int) -> nn.Sequential:
    hidden = min(d, HEAD_WIDTH)
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out))


class ChoiceHead(nn.Module):
    """``none_slot=False`` is the ablation without the *none fits* slot: slot 0 is masked out of the set layer and
    its logit is -inf, so the head is a softmax over the given options only, as an ordinary classifier's is."""

    def __init__(self, d: int, heads: int = 4, none_slot: bool = True):
        super().__init__()
        self.none_slot = none_slot
        self.none = nn.Parameter(torch.zeros(d))
        self.seed = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.score = mlp(d, 1)

    def forward(self, q: torch.Tensor, options: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """q: (n, d); options: (n, K, d); valid: (n, K) bool. Returns (n, 1 + K) logits, slot 0 is *none fits*."""
        slots = torch.cat([(self.none + self.seed(q))[:, None], options], dim=1)
        keep = torch.cat([torch.full_like(valid[:, :1], self.none_slot), valid], dim=1)
        mixed, _ = self.attn(slots, slots, slots, key_padding_mask=~keep, need_weights=False)
        mixed = mixed.nan_to_num(0.0)
        logits = self.score(self.norm(slots + mixed)).squeeze(-1)
        return logits.masked_fill(~keep, float("-inf"))


class YesNoHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = mlp(d, 1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (n, d). Returns (n,) logits of yes."""
        return self.score(q).squeeze(-1)


class ScoreHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.eta = mlp(d, 1)
        self.first = nn.Sequential(nn.Linear(2 * d, min(d, HEAD_WIDTH)), nn.GELU(), nn.Linear(min(d, HEAD_WIDTH), 1))
        self.gap = nn.Sequential(nn.Linear(2 * d, min(d, HEAD_WIDTH)), nn.GELU(), nn.Linear(min(d, HEAD_WIDTH), 1))

    def forward(self, q: torch.Tensor, levels: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q: (n, d); levels: (n, K, d), low to high; valid: (n, K). Returns eta (n,) and cutpoints (n, K - 1),
        increasing along the last axis. Cutpoints past a question's own last level are +inf."""
        pairs = torch.cat([levels[:, :-1], levels[:, 1:]], dim=-1)  # (n, K-1, 2d)
        first = self.first(pairs[:, :1]).squeeze(-1)  # (n, 1)
        gaps = F.softplus(self.gap(pairs[:, 1:]).squeeze(-1)) + 1e-3  # (n, K-2)
        cuts = torch.cat([first, first + torch.cumsum(gaps, dim=-1)], dim=-1)
        cuts = cuts.masked_fill(~valid[:, 1:], float("inf"))
        return self.eta(q).squeeze(-1), cuts


def level_probs(eta: torch.Tensor, cuts: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Cumulative-link probabilities of each level: (n,), (n, K-1) to (n, K). Levels past the scale get 0."""
    cdf = torch.sigmoid((cuts - eta[:, None]) / temperature)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf, torch.ones_like(cdf[:, :1])], dim=-1)
    return (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)
