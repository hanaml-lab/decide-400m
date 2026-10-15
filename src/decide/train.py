"""Training and batched inference over lists of ``Example`` (``toy.py``, or any source of states, questions and
right answers). The full-scale run swaps in ``TreeMaskedModernBert`` and ``HFTokenizer`` (configs/decide-400m.yaml).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import torch

from .model import DecisionModel, Logits, collate, loss
from .packing import pack


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup: float = 0.05
    seed: int = 0


def targets_of(examples: Sequence, where: dict[str, list[tuple[int, int]]]) -> dict[str, torch.Tensor]:
    """The right answers in the order ``Logits.where`` lists the questions. Each example has ``state``,
    ``questions`` and ``answers`` (``data.Record`` or ``toy.Example``)."""
    return {k: torch.tensor([examples[b].answers[i] for b, i in rows], dtype=torch.long) for k, rows in where.items()}


def batches(examples: Sequence, size: int, tok, rng: random.Random | None = None):
    order = list(range(len(examples)))
    if rng:
        rng.shuffle(order)
    for s in range(0, len(order), size):
        chunk = [examples[i] for i in order[s: s + size]]
        packs = [pack(tok, ex.state, ex.questions) for ex in chunk]
        yield chunk, collate(packs, tok.pad_id)


def fit(model: DecisionModel, tok, train: Sequence, cfg: TrainConfig, log=None) -> list[float]:
    """Trains with AdamW, linear warm-up then cosine decay. Returns the mean loss of each epoch."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(train) / cfg.batch_size)
    warm = max(1, int(cfg.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total, count = 0.0, 0
        for chunk, batch in batches(train, cfg.batch_size, tok, rng):
            out = model(batch)
            value = loss(out, targets_of(chunk, out.where))
            opt.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += value.item() * len(chunk)
            count += len(chunk)
        history.append(total / count)
        if log:
            log(f"epoch {epoch + 1}: loss {history[-1]:.3f}")
    return history


@torch.no_grad()
def infer(model: DecisionModel, tok, examples: Sequence, batch_size: int = 128) -> tuple[Logits, dict[str, torch.Tensor]]:
    """Raw logits of every question, by kind, with the right answers in the same order."""
    model.eval()
    parts, targets = [], {"choice": [], "yesno": [], "score": []}
    where: dict[str, list[tuple[int, int]]] = {"choice": [], "yesno": [], "score": []}
    offset = 0
    for chunk, batch in batches(examples, batch_size, tok):
        out = model(batch)
        parts.append(out)
        for k, t in targets_of(chunk, out.where).items():
            targets[k].append(t)
            where[k] += [(offset + b, i) for b, i in out.where[k]]
        offset += len(chunk)
    width = max(p.choice.shape[1] for p in parts)
    cut_width = max(p.cuts.shape[1] for p in parts)
    pad = lambda x, w, v: torch.nn.functional.pad(x, (0, w - x.shape[1]), value=v)  # noqa: E731
    merged = Logits(
        torch.cat([pad(p.choice, width, float("-inf")) for p in parts]),
        torch.cat([p.yesno for p in parts]),
        torch.cat([p.eta for p in parts]),
        torch.cat([pad(p.cuts, cut_width, float("inf")) for p in parts]),
        where,
    )
    return merged, {k: torch.cat(v) for k, v in targets.items()}
