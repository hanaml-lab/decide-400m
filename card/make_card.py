"""The model card's evaluation, as a seeded simulation. Its numbers are illustrative.

decide-400m's weights are not published and its full evaluation has not been run. This script produces the numbers
the card quotes from a simulation whose parameters are set from public zero-shot results on the same datasets and
from the size of effects in the lab's own reports, and it runs the card's actual procedures on the simulated
scores: per-kind temperature scaling fitted on calibration questions, ECE, and the decide-or-defer threshold of
``decide.risk`` set on calibration questions and read on test questions. The procedures are real; the scores they
run on are simulated.

The held-out suite: decision questions built from public classification and NLI datasets that share no dataset
with training.

* choice: Banking77 (77 intents), CLINC150 (150 intents, with its out-of-scope requests asked against intents of
  the closest domain, where no option fits), LEDGAR (100 provision types), TREC (6 question types), AG News (4).
* yes/no: BoolQ, ANLI rounds 1-3 (entailment against contradiction), FEVER (supported against refuted).
* score: SST-5 (5 levels), Yelp-5 (5 stars), STS-B (similarity 0-5, six levels).

    python card/make_card.py           # writes card/results.json
    python card/check_results.py       # checks it against card/results.sha256
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from decide.calibrate import ece, fit_temperature
from decide.risk import decision_threshold, realised

SEED = 20261015
ALPHAS, DELTA = (0.05, 0.10), 0.10
HERE = Path(__file__).resolve().parent

# task: kind, test questions, calibration questions, accuracy of decide-400m and of the NLI baseline (the
# ModernBERT-large zero-shot classifier, one pass per option), all on questions where some option fits.
TASKS = {
    "Banking77": ("choice", 3080, 600, 0.712, 0.574),
    "CLINC150": ("choice", 4500, 600, 0.804, 0.662),
    "LEDGAR": ("choice", 10000, 600, 0.634, 0.493),
    "TREC": ("choice", 500, 300, 0.846, 0.718),
    "AG News": ("choice", 7600, 600, 0.861, 0.792),
    "BoolQ": ("yesno", 3270, 600, 0.792, 0.716),
    "ANLI": ("yesno", 2000, 600, 0.668, 0.604),
    "FEVER": ("yesno", 6000, 600, 0.871, 0.834),
    "SST-5": ("score", 2210, 600, 0.523, 0.412),
    "Yelp-5": ("score", 10000, 600, 0.584, 0.471),
    "STS-B": ("score", 1379, 300, 0.486, 0.334),
}
#: CLINC150's out-of-scope requests: test and calibration questions where no option fits.
NONE_FITS = (1000, 150)
#: How much sharper than right the raw probabilities are, before temperature scaling (the logit is multiplied).
OVERCONFIDENCE = {"choice": 1.55, "yesno": 1.35, "score": 1.7}
HETEROGENEITY = 2.0  # spread of the per-question log-odds of being right


def logit(p):
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(rng, n, accuracy, kind):
    """Per question: whether the answer is right, and its raw confidence."""
    # Per-question log-odds around a centre chosen so the mean probability of being right is ``accuracy``.
    eps = rng.standard_normal(n) * HETEROGENEITY
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sigmoid(mid + eps).mean() < accuracy else (lo, mid)
    p = sigmoid((lo + hi) / 2 + eps)
    right = rng.uniform(size=n) < p
    # A top-label confidence cannot sit below chance for the kind; the raw score is overconfident.
    raw = sigmoid(OVERCONFIDENCE[kind] * logit(np.clip(p, 1e-4, 1 - 1e-4)))
    return right, raw


def temperature(conf, right):
    """The temperature that best calibrates top-label confidences (binary NLL on the logit)."""
    z = logit(np.clip(conf, 1e-6, 1 - 1e-6))
    def nll(t):
        q = np.clip(sigmoid(z / t), 1e-9, 1 - 1e-9)
        return float(-(right * np.log(q) + (~right) * np.log1p(-q)).mean())
    return fit_temperature(nll)


def main() -> dict:
    rng = np.random.default_rng(SEED)
    data = {}
    for name, (kind, n_test, n_cal, acc, base) in TASKS.items():
        r_cal, c_cal = simulate(rng, n_cal, acc, kind)
        r_test, c_test = simulate(rng, n_test, acc, kind)
        b_test, _ = simulate(rng, n_test, base, kind)
        data[name] = dict(kind=kind, cal=(r_cal, c_cal), test=(r_test, c_test), base_acc=float(b_test.mean()))
    # Out of scope: decide-400m puts its mass on "none fits", so its top option is weak; the baseline's softmax
    # over the given options must pick one, and every pick is wrong.
    oos_cal = sigmoid(rng.normal(-1.5, 1.2, NONE_FITS[1]))
    oos_test = sigmoid(rng.normal(-1.5, 1.2, NONE_FITS[0]))

    res: dict = {"seed": SEED, "alphas": list(ALPHAS), "delta": DELTA, "tasks": {}, "kinds": {}, "decisions": {}}
    for name, d in data.items():
        res["tasks"][name] = {"kind": d["kind"], "test_questions": len(d["test"][0]),
                              "accuracy": round(float(d["test"][0].mean()), 3), "baseline_accuracy": round(d["base_acc"], 3)}
    res["macro_accuracy"] = round(float(np.mean([t["accuracy"] for t in res["tasks"].values()])), 3)
    res["macro_baseline_accuracy"] = round(float(np.mean([t["baseline_accuracy"] for t in res["tasks"].values()])), 3)

    pooled = {a: ([], []) for a in ALPHAS}
    for kind in ("choice", "yesno", "score"):
        names = [n for n, d in data.items() if d["kind"] == kind]
        r_cal = np.concatenate([data[n]["cal"][0] for n in names])
        c_cal = np.concatenate([data[n]["cal"][1] for n in names])
        r_test = np.concatenate([data[n]["test"][0] for n in names])
        c_test = np.concatenate([data[n]["test"][1] for n in names])
        if kind == "choice":  # questions where no option fits: any decision is wrong
            r_cal, c_cal = np.concatenate([r_cal, np.zeros(len(oos_cal), bool)]), np.concatenate([c_cal, oos_cal])
            r_test, c_test = np.concatenate([r_test, np.zeros(len(oos_test), bool)]), np.concatenate([c_test, oos_test])
        t = temperature(c_cal, r_cal)
        cal_conf = sigmoid(logit(np.clip(c_cal, 1e-6, 1 - 1e-6)) / t)
        test_conf = sigmoid(logit(np.clip(c_test, 1e-6, 1 - 1e-6)) / t)
        res["kinds"][kind] = {"temperature": round(t, 3), "ece_raw": round(ece(c_test, r_test), 3),
                              "ece_calibrated": round(ece(test_conf, r_test), 3)}
        for a in ALPHAS:
            th = decision_threshold(cal_conf, ~r_cal, alpha=a, delta=DELTA)
            cov, err = realised(test_conf, ~r_test, th.value)
            res["decisions"].setdefault(f"{a:.2f}", {})[kind] = {
                "threshold": round(th.value, 4) if math.isfinite(th.value) else None, "coverage": round(cov, 3), "error_among_decided": round(err, 4)}
            decided = test_conf >= th.value
            pooled[a][0].append(decided)
            pooled[a][1].append(~r_test & decided)
            if kind == "choice":
                res["decisions"][f"{a:.2f}"]["none_fits_decided"] = round(float(decided[-len(oos_test):].mean()), 4)
    for a in ALPHAS:
        decided, wrong = np.concatenate(pooled[a][0]), np.concatenate(pooled[a][1])
        res["decisions"][f"{a:.2f}"]["coverage"] = round(float(decided.mean()), 3)
        res["decisions"][f"{a:.2f}"]["error_among_decided"] = round(float(wrong.sum() / max(decided.sum(), 1)), 4)
    res["ece_raw_mean"] = round(float(np.mean([k["ece_raw"] for k in res["kinds"].values()])), 3)
    res["ece_calibrated_mean"] = round(float(np.mean([k["ece_calibrated"] for k in res["kinds"].values()])), 3)
    (HERE / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: v for k, v in out.items() if k != "tasks"}, indent=1))
    assert math.isfinite(out["macro_accuracy"])
