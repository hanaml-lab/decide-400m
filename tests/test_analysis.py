"""The report's analysis: its vectorised threshold is the shipped one, and its probabilities match the model's."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analyze = load("analyze")
evaluate = load("evaluate")


def test_threshold_is_decision_threshold():
    from decide.risk import decision_threshold

    rng = np.random.default_rng(0)
    for n, noise in ((300, 0.5), (2000, 1.0), (5000, 2.0)):
        conf = rng.uniform(0.3, 1.0, n)
        wrong = rng.uniform(size=n) > conf - rng.uniform(0, noise * 0.1, n)
        for a in (0.02, 0.05, 0.1, 0.2):
            assert analyze.threshold(conf, wrong, a) == decision_threshold(conf, wrong, a, 0.1).value


def rows(kind, logits, targets, form):
    return [{"r": i, "q": 0, "task": "t", "kind": kind, "tag": "natural", "target": t, "form": form, "k": len(z) - 1 if kind != "yesno" else 0,
             "logits": z} for i, (z, t) in enumerate(zip(logits, targets))]


def test_kind_probabilities_match_the_heads():
    from decide.heads import level_probs

    eta, cuts = 0.3, [-1.0, 0.2, 1.5]
    k = analyze.Kind(rows("score", [[eta] + cuts, [eta] + cuts[:1]], [2, 1], "cumlink"))
    p = k.probs(1.7)
    ref = level_probs(torch.tensor([eta]), torch.tensor([cuts]), 1.7)[0].numpy()
    assert np.allclose(p[0], ref, atol=1e-6) and np.allclose(p.sum(1), 1)
    assert p[1, 2:].sum() == 0  # a two-level scale has no third level
    c = analyze.Kind(rows("choice", [[0.5, 2.0, -1.0], [3.0, 0.0, 0.1]], [1, 0], "slots"))
    ans, conf, ok = c.decisions(1.0)
    assert list(ans) == [0, 1] and ok[0] and not ok[1]  # a choice no option fits is wrong when decided
    y = analyze.Kind(rows("yesno", [[1.2], [-0.4]], [1, 1], "logit"))
    ans, conf, ok = y.decisions(1.0)
    assert list(ok) == [True, False] and conf[1] == pytest.approx(1 - 1 / (1 + np.exp(0.4)))


def test_no_slot_models_fit_temperatures_on_fitting_questions():
    c = analyze.Kind(rows("choice", [[None, 2.0, -1.0], [None, 0.0, 0.1], [None, 1.0, 0.0]], [1, 0, 2], "slots"))
    assert list(c.fit_mask()) == [True, False, True]
    assert np.isfinite(analyze.fit_t(c))


def test_invariance_helper():
    base = [{"record": 0, "question": 0, "kind": "choice", "logits": [0.1, 1.0, 2.0, 3.0]}]
    perms = [{0: [2, 0, 1]}]
    moved = [{"record": 0, "question": 0, "kind": "choice", "logits": [0.1, 3.0, 1.0, 2.0]}]
    got = evaluate.invariance(base, moved, perms)
    assert got["max_option_prob_change"] < 1e-12 and got["answer_changed"] == 0
