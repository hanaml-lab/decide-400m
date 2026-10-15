"""Calibration: one temperature per answer kind, fitted on held-out questions by negative log-likelihood, and the
expected calibration error (ECE) of the result.

A temperature T divides a choice's logits, a yes/no logit, or the gaps (theta_j - eta) of a score's cumulative
link. One parameter per kind keeps the ranking of answers unchanged (Guo et al., 2017), so accuracy does not move;
only the stated probabilities do.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .heads import level_probs
from .model import Logits

KINDS = ("choice", "yesno", "score")


def nll(out: Logits, targets: dict[str, torch.Tensor], kind: str, temperature: float) -> float:
    """Mean negative log-likelihood of the right answers of one kind at a temperature."""
    with torch.no_grad():
        if kind == "choice":
            return torch.nn.functional.cross_entropy(out.choice / temperature, targets["choice"]).item()
        if kind == "yesno":
            return torch.nn.functional.binary_cross_entropy_with_logits(out.yesno / temperature, targets["yesno"].float()).item()
        p = level_probs(out.eta, out.cuts, temperature).gather(1, targets["score"][:, None])
        return (-torch.log(p.clamp_min(1e-12))).mean().item()


def fit_temperature(loss_at, lo: float = 0.05, hi: float = 20.0, steps: int = 60) -> float:
    """Minimises ``loss_at(T)`` over log T by golden-section search. NLL is unimodal in log T for these heads."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss_at(math.exp(c)), loss_at(math.exp(d))
    for _ in range(steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss_at(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss_at(math.exp(d))
    return math.exp((a + b) / 2)


def fit_temperatures(out: Logits, targets: dict[str, torch.Tensor]) -> dict[str, float]:
    """One temperature per kind present in ``out``."""
    present = {"choice": len(out.choice), "yesno": len(out.yesno), "score": len(out.eta)}
    return {k: fit_temperature(lambda t, k=k: nll(out, targets, k, t)) for k in KINDS if present[k]}


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins (Guo et al., 2017): the weighted mean, over bins, of
    |accuracy - mean confidence|."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = which == b
        if sel.any():
            total += sel.mean() * abs(correct[sel].mean() - confidence[sel].mean())
    return float(total)


def top_label(out: Logits, targets: dict[str, torch.Tensor], kind: str, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """Confidence and correctness of each question's most probable answer, for ECE. For a choice the most probable
    answer may be *none fits*."""
    with torch.no_grad():
        if kind == "choice":
            p = torch.softmax(out.choice / temperature, dim=-1)
            conf, pick = p.max(dim=-1)
            return conf.numpy(), (pick == targets["choice"]).numpy()
        if kind == "yesno":
            p = torch.sigmoid(out.yesno / temperature)
            return torch.maximum(p, 1 - p).numpy(), ((p >= 0.5).long() == targets["yesno"]).numpy()
        p = level_probs(out.eta, out.cuts, temperature)
        conf, pick = p.max(dim=-1)
        return conf.numpy(), (pick == targets["score"]).numpy()


@dataclass
class Calibration:
    """What a deployment needs besides the weights: a temperature per kind, and the confidence each kind must
    reach before an answer is decided rather than deferred (``risk.py``), for the error level ``alpha`` and the
    failure probability ``delta`` they were set for."""

    temperatures: dict[str, float]
    thresholds: dict[str, float] = field(default_factory=dict)
    alpha: float = 0.05
    delta: float = 0.1
    calibration_questions: dict[str, int] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text()))
