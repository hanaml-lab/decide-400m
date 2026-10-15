import math

import numpy as np

from decide.risk import GRID, binom_cdf, decision_threshold, realised, state_threshold


def test_binom_cdf():
    assert abs(binom_cdf(0, 10, 0.1) - 0.9**10) < 1e-12
    exact = sum(math.comb(20, i) * 0.3**i * 0.7 ** (20 - i) for i in range(5))
    assert abs(binom_cdf(4, 20, 0.3) - exact) < 1e-12
    assert binom_cdf(20, 20, 0.3) == 1.0


def synthetic(rng, n):
    conf = rng.uniform(0.5, 1.0, n)
    wrong = rng.uniform(size=n) > conf ** 3  # errors fall as confidence rises
    return conf, wrong


def true_rate(threshold):
    # P(wrong | conf >= t) for conf ~ U(0.5, 1) and P(wrong | c) = 1 - c^3.
    if threshold == float("inf"):
        return 0.0
    t = max(threshold, 0.5)
    return 1 - (1 - t**4) / (4 * (1 - t))


def test_the_bound_holds_with_probability_one_minus_delta():
    rng = np.random.default_rng(0)
    alpha, delta, fails = 0.1, 0.1, 0
    for _ in range(300):
        th = decision_threshold(*synthetic(rng, 1000), alpha=alpha, delta=delta)
        fails += true_rate(th.value) > alpha
    assert fails / 300 <= delta


def test_nothing_is_decided_when_nothing_can_be_certified():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.5, 1, 500)
    th = decision_threshold(conf, np.ones(500, bool), alpha=0.1)
    assert th.value == float("inf") and th.decided == 0
    assert realised(conf, np.ones(500, bool), th.value) == (0.0, 0.0)


def test_state_threshold_controls_the_expected_loss():
    rng = np.random.default_rng(2)
    alpha, losses_new = 0.1, []
    for _ in range(200):
        c = rng.uniform(0.5, 1, (400, 3))
        w = rng.uniform(size=c.shape) > c**3
        loss = np.stack([((c >= t) & w).any(axis=1) for t in GRID], axis=1).astype(float)
        th = state_threshold(loss, alpha)
        c2 = rng.uniform(0.5, 1, 3)
        losses_new.append(float(((c2 >= th) & (rng.uniform(size=3) > c2**3)).any()))
    assert np.mean(losses_new) <= alpha + 0.03
