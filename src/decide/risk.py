"""Deciding or deferring with a bound on the error rate of the decisions made.

``decision_threshold`` (question level). An answer is decided when its calibrated confidence reaches a threshold
lambda, and deferred otherwise. The quantity controlled is the *selective* error rate R(lambda) = P(wrong |
decided), the false-finish rate of CCV's submit rule applied to any answer. For each lambda on a grid fixed in
advance, the calibration questions it would decide give n decisions with e errors; under exchangeability e is
Binomial(n, R(lambda)) given n, so p = P(Binomial(n, alpha) <= e) is a valid p-value for the hypothesis R(lambda) >
alpha. Testing every grid point at delta / m (Bonferroni, one of the family-wise procedures of Learn then Test)
and taking the lowest threshold whose hypothesis is rejected gives

    P( R(lambda_hat) <= alpha ) >= 1 - delta   over the draw of the calibration questions.

CCV's rule bounds the same rate in expectation, up to O(n^-1/2); this one holds with probability 1 - delta and is
more conservative for it. With no threshold rejected, everything is deferred.

``state_threshold`` (state level). InForce's answer-level guarantee: the loss of a state is 1 if any decision made
about it is wrong. It does not increase as the threshold rises, so conformal risk control gives E[loss] <= alpha
on a new state, in expectation over the calibration draw.

Both assume the calibration questions are exchangeable with the ones asked later. Under shift they are not, and
both papers show the bound failing; recalibrate on questions from the new distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The thresholds tested, fixed before any data is seen.
GRID = np.round(np.linspace(0.50, 0.995, 100), 4)


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= k), summed in log space."""
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)
            for i in range(k + 1)]
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(v - top) for v in logs))


@dataclass
class Threshold:
    """The chosen threshold (``inf`` when every answer must be deferred) and what it did on calibration."""

    value: float
    decided: int
    errors: int
    n: int

    @property
    def coverage(self) -> float:
        return self.decided / self.n if self.n else 0.0


def decision_threshold(confidence, wrong, alpha: float, delta: float = 0.1, grid=GRID) -> Threshold:
    """The lowest grid threshold at which the selective error rate is certified at most ``alpha``."""
    confidence, wrong = np.asarray(confidence, float), np.asarray(wrong, bool)
    level = delta / len(grid)
    best = Threshold(float("inf"), 0, 0, len(confidence))
    for lam in sorted(grid, reverse=True):
        decided = confidence >= lam
        n, e = int(decided.sum()), int(wrong[decided].sum())
        if n and binom_cdf(e, n, alpha) <= level:
            best = Threshold(float(lam), n, e, len(confidence))
    return best


def state_threshold(losses: np.ndarray, alpha: float, grid=GRID, bound: float = 1.0) -> float:
    """Conformal risk control over states. ``losses[i, j]`` is the loss of calibration state i at ``grid[j]``, in
    [0, bound] and non-increasing in the threshold. Returns the lowest threshold with
    (n / (n + 1)) * mean loss + bound / (n + 1) <= alpha, or ``inf``."""
    losses = np.asarray(losses, float)
    n = losses.shape[0]
    order = np.argsort(grid)
    for j in order:
        if (n / (n + 1)) * losses[:, j].mean() + bound / (n + 1) <= alpha:
            return float(grid[j])
    return float("inf")


def realised(confidence, wrong, threshold: float) -> tuple[float, float]:
    """Coverage (share decided) and error rate among the decided, on new questions."""
    confidence, wrong = np.asarray(confidence, float), np.asarray(wrong, bool)
    decided = confidence >= threshold
    if not decided.any():
        return 0.0, 0.0
    return float(decided.mean()), float(wrong[decided].mean())
