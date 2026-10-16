"""Every number of the report, computed from the committed prediction files.

    python scripts/analyze.py --results results --out results/summary.json

Reads ``results/<run>/predictions_test_{in,out}.jsonl.gz`` (``scripts/evaluate.py``, ``scripts/baseline_nli.py``) and
writes one JSON file. Nothing here touches a model: the analysis can be rerun, and checked, from the predictions
alone. Conventions (the report's Section 4):

* The answer to a choice is its most probable *option* and its confidence that option's probability, so mass on
  *none fits* lowers it; a question no option fits is wrong whenever it is decided. A yes/no answers yes at p >= 0.5
  with confidence max(p, 1 - p); a score answers its most probable level.
* Accuracy is of that answer: on choice questions some option fits, on every yes/no and score question.
* Temperatures are fitted by negative log-likelihood, one per kind (or one for all kinds), on the ``dev`` states:
  training-task states held back from training and in neither suite. The confidence score is therefore fixed before
  any calibration question is seen, as Learn then Test requires. ECE uses 15 equal-width bins of the decision's
  confidence, over every question of the kind; it is also reported with a temperature refitted on each suite's
  calibration half, for reference.
* Decide or defer: ``decide.risk.decision_threshold`` (Learn then Test, binomial tests at delta / 100 over a grid
  of 100 thresholds), per kind, at delta = 0.1. Calibration and test questions are a random half each of a suite's
  states; this is repeated over ``--splits`` random splits, and the report gives the mean coverage, the mean error
  among decided questions and the share of splits whose error exceeds alpha (which the guarantee bounds by delta).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit, log_softmax
from scipy.stats import binom, rankdata

from decide.calibrate import fit_temperature
from decide.risk import GRID, decision_threshold

KINDS = ("choice", "yesno", "score")
ALPHAS = (0.02, 0.05, 0.10, 0.15, 0.20)
DELTA = 0.1
BINS = 15
HELD_OUT = ["trec", "banking77", "newsgroups", "rte", "scitail", "mrpc", "sst5", "stsb", "tweet_sentiment"]
IN_DOMAIN = ["clinc", "ag_news", "dbpedia", "yahoo", "mnli", "boolq", "yelp"]
BASELINE_SEEN = {"banking77", "ag_news", "yahoo", "yelp", "mnli", "sst5"}
NAMES = {"clinc": "CLINC150", "ag_news": "AG News", "dbpedia": "DBpedia-14", "yahoo": "Yahoo Answers",
         "mnli": "MultiNLI", "boolq": "BoolQ", "yelp": "Yelp-5", "trec": "TREC-6", "banking77": "Banking77",
         "newsgroups": "20 Newsgroups", "rte": "RTE", "scitail": "SciTail", "mrpc": "MRPC", "sst5": "SST-5",
         "stsb": "STS-B", "tweet_sentiment": "TweetEval sentiment"}


# ------------------------------------------------------------------------------------------------- loading

class Kind:
    """The questions of one kind in one suite, as arrays: everything the analysis needs from the logits."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.n = len(rows)
        self.state = np.array([r["r"] for r in rows])
        self.task = np.array([r["task"] for r in rows])
        self.tag = np.array([r["tag"] for r in rows])
        self.target = np.array([r["target"] for r in rows])
        self.form = rows[0]["form"] if rows else ""
        width = max((len(r["logits"]) for r in rows), default=1)
        z = np.full((self.n, width), -np.inf)
        for i, r in enumerate(rows):
            vals = [(-np.inf if v is None else v) for v in r["logits"]]
            z[i, : len(vals)] = vals
        self.z = z
        self.k = np.array([r["k"] for r in rows])

    def subset(self, mask: np.ndarray) -> "Kind":
        return Kind([r for r, m in zip(self.rows, mask) if m])

    # probabilities of every slot/level at temperature t
    def probs(self, t: float) -> np.ndarray:
        if self.form == "slots" or self.form == "softmax":
            return np.exp(log_softmax(self.z / t, axis=1))
        if self.form == "logit":
            p = expit(self.z[:, 0] / t)
            return np.stack([1 - p, p], axis=1)
        eta, cuts = self.z[:, :1], self.z[:, 1:]  # cumulative link; cuts past a scale are -inf here, +inf meant
        cuts = np.where(np.isfinite(cuts), cuts, np.inf)
        cdf = expit((cuts - eta) / t)
        cdf = np.concatenate([np.zeros((self.n, 1)), cdf, np.ones((self.n, 1))], axis=1)
        p = np.clip(np.diff(cdf, axis=1), 0, None)
        return p

    def nll(self, t: float, mask=None) -> float:
        p = self.probs(t)
        idx = self.target
        if self.form == "logit":
            got = p[np.arange(self.n), idx]
        else:
            got = p[np.arange(self.n), idx]
        if mask is not None:
            got = got[mask]
        return float(-np.log(np.clip(got, 1e-12, None)).mean())

    def fit_mask(self) -> np.ndarray:
        """Questions a temperature can be fitted on: all, except a choice no option fits when the model has no
        none-fits slot (its likelihood is 0 at any temperature)."""
        if self.form == "slots" and not np.isfinite(self.z[:, 0]).any():
            return self.target > 0
        return np.ones(self.n, bool)

    def decisions(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(answer, confidence, correct) of every question at temperature t."""
        p = self.probs(t)
        if self.form == "slots":
            opts = p[:, 1:]
            ans = opts.argmax(1)
            conf = opts[np.arange(self.n), ans]
            return ans, conf, (ans + 1) == self.target
        if self.form == "logit":
            ans = (p[:, 1] >= 0.5).astype(int)
            return ans, p.max(1), ans == self.target
        ans = p.argmax(1)
        return ans, p[np.arange(self.n), ans], ans == self.target


def load(path: Path) -> dict[str, Kind]:
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    return {k: Kind([r for r in rows if r["kind"] == k]) for k in KINDS}


# ------------------------------------------------------------------------------------------------- metrics

def fit_t(kind: Kind, mask=None) -> float:
    m = kind.fit_mask() if mask is None else (mask & kind.fit_mask())
    if not m.any():
        return 1.0
    return fit_temperature(lambda t: kind.nll(t, m))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = which == b
        if sel.any():
            total += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(total)


def reliability(conf, correct, bins: int = BINS) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    out = []
    for b in range(bins):
        sel = which == b
        if sel.any():
            out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(sel.sum()),
                        "confidence": float(conf[sel].mean()), "accuracy": float(correct[sel].mean())})
    return out


def threshold(conf: np.ndarray, wrong: np.ndarray, alpha: float, delta: float = DELTA) -> float:
    """``decide.risk.decision_threshold``, vectorised over the grid (checked against it in ``check_threshold``)."""
    level = delta / len(GRID)
    decided = conf[None, :] >= GRID[:, None]
    n = decided.sum(1)
    e = (decided & wrong[None, :]).sum(1)
    p = np.where(n > 0, binom.cdf(e, np.maximum(n, 1), alpha), 1.0)
    ok = np.where((n > 0) & (p <= level))[0]
    return float(GRID[ok].min()) if len(ok) else float("inf")


def check_threshold(conf, wrong) -> None:
    for a in (0.05, 0.10):
        assert threshold(conf, wrong, a) == decision_threshold(conf, wrong, a, DELTA).value


def realised(conf, wrong, th) -> tuple[float, float]:
    d = conf >= th
    if not d.any():
        return 0.0, 0.0
    return float(d.mean()), float(wrong[d].mean())


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = positive.sum(), (~positive).sum()
    if not pos or not neg:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - pos * (pos + 1) / 2) / (pos * neg))


def split_states(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random half of the suite's states: True for calibration."""
    uniq = np.unique(states)
    cal = set(rng.choice(uniq, size=len(uniq) // 2, replace=False).tolist())
    return np.array([s in cal for s in states])


# ------------------------------------------------------------------------------------------------- per run

def accuracy_table(kinds: dict[str, Kind], tasks: list[str]) -> dict:
    """Accuracy at temperature 1 (temperatures do not change the answer), by kind and by task and tag."""
    out = {"by_kind": {}, "by_task": {}}
    for k, kind in kinds.items():
        if not kind.n:
            continue
        _, conf, correct = kind.decisions(1.0)
        fits = kind.target > 0 if k == "choice" else np.ones(kind.n, bool)
        nat = kind.tag == "natural"
        out["by_kind"][k] = {"accuracy": float(correct[fits].mean()), "questions": int(fits.sum()),
                             "natural_accuracy": float(correct[fits & nat].mean()) if (fits & nat).any() else None,
                             "natural_questions": int((fits & nat).sum())}
        if k == "choice":
            out["by_kind"][k]["none_fits_questions"] = int((~fits).sum())
            if np.isfinite(kind.z[:, 0]).any() and (~fits).any():
                p_none = kind.probs(1.0)[:, 0]
                out["by_kind"][k]["none_fits_auroc"] = auroc(p_none, ~fits)
                # how much probability the slot takes from questions that do have a right option
                out["by_kind"][k]["none_mass_on_fitting"] = float(p_none[fits].mean())
            elif (~fits).any():
                out["by_kind"][k]["none_fits_auroc"] = auroc(-conf, ~fits)
        if k == "score":
            ans, _, _ = kind.decisions(1.0)
            p = kind.probs(1.0)
            expected = (p * np.arange(p.shape[1])[None]).sum(1)
            out["by_kind"][k]["mae_levels"] = float(np.abs(ans - kind.target).mean())
            out["by_kind"][k]["within_one"] = float((np.abs(ans - kind.target) <= 1).mean())
            out["by_kind"][k]["spearman_expected"] = float(np.corrcoef(rankdata(expected), rankdata(kind.target))[0, 1])
        for task in tasks:
            for tag in ("natural", "derived"):
                sel = (kind.task == task) & (kind.tag == tag) & fits
                if sel.any():
                    row = {"kind": k, "tag": tag, "accuracy": float(correct[sel].mean()), "questions": int(sel.sum())}
                    # chance: a uniform pick among a choice's options; the commonest answer for yes/no and score
                    if k == "choice":
                        row["chance"] = float(np.mean(1.0 / kind.k[sel]))
                    else:
                        row["chance"] = float(np.bincount(kind.target[sel]).max() / sel.sum())
                    if k == "score":
                        row["mae_levels"] = float(np.abs(ans[sel] - kind.target[sel]).mean())
                    out["by_task"].setdefault(task, []).append(row)
    return out


def single_temperature(kinds: dict[str, Kind], masks: dict[str, np.ndarray] | None = None) -> float:
    """One temperature for every kind, fitted on the pooled likelihood."""
    fit = {k: kinds[k].fit_mask() & (masks[k] if masks else True) for k in kinds if kinds[k].n}
    return fit_temperature(lambda t: sum(kinds[k].nll(t, fit[k]) * fit[k].sum() for k in fit)
                           / sum(m.sum() for m in fit.values()))


def dev_temperatures(dev: dict[str, Kind]) -> dict:
    temps = {k: fit_t(dev[k]) for k in KINDS if dev[k].n}
    return {"per_kind": temps, "single": single_temperature(dev)}


def calibration_table(kinds: dict[str, Kind], cal: dict[str, np.ndarray], temps: dict) -> dict:
    """ECE and NLL on the suite's test half: raw, with the dev temperatures (one per kind, or one for all), and
    with temperatures refitted on the suite's calibration half (one per kind, or one for all)."""
    out = {}
    t_suite_single = single_temperature(kinds, cal)
    for k, kind in kinds.items():
        if not kind.n:
            continue
        t_dev = temps["per_kind"].get(k, 1.0)
        t_suite = fit_t(kind, cal[k])
        test = ~cal[k]
        row = {"temperature_dev": t_dev, "temperature_single": temps["single"], "temperature_suite": t_suite,
               "temperature_suite_single": t_suite_single}
        for name, temp in (("raw", 1.0), ("calibrated", t_dev), ("single", temps["single"]), ("suite", t_suite),
                           ("suite_single", t_suite_single)):
            _, conf, correct = kind.decisions(temp)
            row[f"ece_{name}"] = ece(conf[test], correct[test])
            row[f"nll_{name}"] = kind.nll(temp, kind.fit_mask() & test)
        _, c1, r1 = kind.decisions(1.0)
        _, ct, rt = kind.decisions(t_dev)
        _, cs, rs = kind.decisions(t_suite)
        row["reliability_raw"] = reliability(c1[test], r1[test])
        row["reliability_calibrated"] = reliability(ct[test], rt[test])
        row["reliability_suite"] = reliability(cs[test], rs[test])
        out[k] = row
    return out


def decide_table(kinds: dict[str, Kind], temps: dict, splits: int, seed: int, alphas=ALPHAS, natural_only=False) -> dict:
    """Decide or defer over random calibration/test splits of the suite's states, at the dev temperatures."""
    rng = np.random.default_rng(seed)
    # natural_only: the natural questions the NLI baseline also answers (it does not answer BoolQ), so the two models
    # are compared on the same questions
    use = {k: ((kinds[k].tag == "natural") & (kinds[k].task != "boolq")) if natural_only else np.ones(kinds[k].n, bool)
           for k in kinds}
    states = np.unique(np.concatenate([kinds[k].state[use[k]] for k in kinds if kinds[k].n]))
    acc = {a: {k: {"coverage": [], "error": [], "violations": 0, "none_decided": []} for k in KINDS} for a in alphas}
    pooled = {a: {"coverage": [], "error": [], "violations": 0} for a in alphas}
    for s in range(splits):
        cal_states = set(rng.choice(states, size=len(states) // 2, replace=False).tolist())
        tot_d, tot_w, tot_n = {a: 0 for a in alphas}, {a: 0 for a in alphas}, {a: 0 for a in alphas}
        for k, kind in kinds.items():
            if not kind.n or not use[k].any():
                continue
            cal = np.array([x in cal_states for x in kind.state]) & use[k]
            test = ~np.array([x in cal_states for x in kind.state]) & use[k]
            _, conf, correct = kind.decisions(temps["per_kind"].get(k, 1.0))
            wrong = ~correct
            if s == 0:
                check_threshold(conf[cal], wrong[cal])
            for a in alphas:
                th = threshold(conf[cal], wrong[cal], a)
                cov, err = realised(conf[test], wrong[test], th)
                cell = acc[a][k]
                cell["coverage"].append(cov)
                cell["error"].append(err)
                cell["violations"] += int(err > a)
                if k == "choice" and (kind.target[test] == 0).any():
                    cell["none_decided"].append(float((conf[test & (kind.target == 0)] >= th).mean()))
                d = conf[test] >= th
                tot_d[a] += int(d.sum())
                tot_w[a] += int((wrong[test] & d).sum())
                tot_n[a] += int(test.sum())
        for a in alphas:
            pooled[a]["coverage"].append(tot_d[a] / max(tot_n[a], 1))
            e = tot_w[a] / max(tot_d[a], 1)
            pooled[a]["error"].append(e)
            pooled[a]["violations"] += int(e > a)
    out = {}
    for a in alphas:
        row = {}
        for k in KINDS:
            c = acc[a][k]
            if not c["coverage"]:
                continue
            row[k] = {"coverage": float(np.mean(c["coverage"])), "error": float(np.mean(c["error"])),
                      "error_p90": float(np.quantile(c["error"], 0.9)), "violation_rate": c["violations"] / splits,
                      "coverage_sd": float(np.std(c["coverage"]))}
            if c["none_decided"]:
                row[k]["none_fits_decided"] = float(np.mean(c["none_decided"]))
        row["pooled"] = {"coverage": float(np.mean(pooled[a]["coverage"])), "error": float(np.mean(pooled[a]["error"])),
                         "violation_rate": pooled[a]["violations"] / splits}
        out[f"{a:.2f}"] = row
    return out


def shift_table(src: dict[str, Kind], dst: dict[str, Kind], temps: dict, seed: int,
                sizes=(50, 100, 200, 500, 1000, 2000, 3000, 5000),
                splits: int = 200, alphas=(0.05, 0.10)) -> dict:
    """Thresholds set on the whole in-domain suite and read on the held-out one; then thresholds set on n held-out
    questions of each kind, tested on the rest. Temperatures are the dev ones throughout."""
    out = {"transfer": {}, "recalibrated": {}}
    for k in KINDS:
        if not src[k].n or not dst[k].n:
            continue
        t = temps["per_kind"].get(k, 1.0)
        _, conf_s, corr_s = src[k].decisions(t)
        _, conf_d, corr_d = dst[k].decisions(t)
        for a in ALPHAS:  # carrying thresholds over is cheap: every level
            th = threshold(conf_s, ~corr_s, a)
            cov, err = realised(conf_d, ~corr_d, th)
            row = {"threshold": th, "coverage": cov, "error": err}
            by_task = {}
            for task in np.unique(dst[k].task):
                sel = dst[k].task == task
                c2, e2 = realised(conf_d[sel], ~corr_d[sel], th)
                by_task[task] = {"coverage": c2, "error": e2, "questions": int(sel.sum())}
            row["by_task"] = by_task
            out["transfer"].setdefault(f"{a:.2f}", {})[k] = row
    rng = np.random.default_rng(seed)
    for k in KINDS:
        kind = dst[k]
        if not kind.n:
            continue
        _, conf, corr = kind.decisions(temps["per_kind"].get(k, 1.0))
        for n in sizes:
            if n >= kind.n // 2:
                continue
            for a in alphas:
                covs, errs, viol = [], [], 0
                for _ in range(splits):
                    cal = np.zeros(kind.n, bool)
                    cal[rng.choice(kind.n, size=n, replace=False)] = True
                    th = threshold(conf[cal], ~corr[cal], a)
                    cov, err = realised(conf[~cal], ~corr[~cal], th)
                    covs.append(cov)
                    errs.append(err)
                    viol += int(err > a)
                out["recalibrated"].setdefault(f"{a:.2f}", {}).setdefault(k, {})[str(n)] = {
                    "coverage": float(np.mean(covs)), "error": float(np.mean(errs)), "violation_rate": viol / splits}
    return out


def per_task_conditional(kinds: dict[str, Kind], temps: dict, seed: int, alpha: float = 0.10, splits: int = 200) -> dict:
    """Thresholds pooled over a kind's tasks guarantee the pooled error, not each task's. Error by task."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, kind in kinds.items():
        if not kind.n:
            continue
        tasks = np.unique(kind.task)
        errs = {t: [] for t in tasks}
        covs = {t: [] for t in tasks}
        _, conf, corr = kind.decisions(temps["per_kind"].get(k, 1.0))
        for _ in range(splits):
            cal = split_states(kind.state, rng)
            th = threshold(conf[cal], ~corr[cal], alpha)
            for task in tasks:
                sel = (~cal) & (kind.task == task)
                c, e = realised(conf[sel], ~corr[sel], th)
                covs[task].append(c)
                errs[task].append(e)
        out[k] = {task: {"coverage": float(np.mean(covs[task])), "error": float(np.mean(errs[task])),
                         "violation_rate": float(np.mean(np.array(errs[task]) > alpha))} for task in tasks}
    return out


def analyze_run(path: Path, splits: int, seed: int, full: bool) -> dict:
    res = {}
    suites = {s: load(path / f"predictions_{s}.jsonl.gz") for s in ("test_in", "test_out")
              if (path / f"predictions_{s}.jsonl.gz").exists()}
    temps = dev_temperatures(load(path / "predictions_dev.jsonl.gz"))
    res["temperatures"] = temps
    rng = np.random.default_rng(seed)
    for name, kinds in suites.items():
        tasks = IN_DOMAIN if name == "test_in" else HELD_OUT
        states = np.unique(np.concatenate([kinds[k].state for k in kinds if kinds[k].n]))
        cal_states = set(rng.choice(states, size=len(states) // 2, replace=False).tolist())
        cal = {k: np.array([x in cal_states for x in kinds[k].state]) for k in kinds}
        res[name] = {"accuracy": accuracy_table(kinds, tasks), "calibration": calibration_table(kinds, cal, temps),
                     "decide": decide_table(kinds, temps, splits, seed)}
        if full:
            res[name]["decide_natural"] = decide_table(kinds, temps, splits, seed, natural_only=True)
            res[name]["per_task_conditional"] = per_task_conditional(kinds, temps, seed, splits=splits)
    if full and len(suites) == 2:
        res["shift"] = shift_table(suites["test_in"], suites["test_out"], temps, seed, splits=splits)
    for f in ("invariance.json", "eval_summary.json", "train_summary.json", "config.json"):
        if (path / f).exists():
            res[f.replace(".json", "")] = json.loads((path / f).read_text())
    return res


def macro(acc: dict, tasks: list[str], exclude=()) -> float | None:
    vals = [row["accuracy"] for t in tasks if t not in exclude for row in acc["by_task"].get(t, [])
            if row["tag"] == "natural"]
    return float(np.mean(vals)) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.json")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    root = Path(args.results)
    out: dict = {"splits": args.splits, "delta": DELTA, "alphas": list(ALPHAS), "runs": {}}
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p / "predictions_test_out.jsonl.gz").exists()):
        full = run.name.startswith("decide-400m-s") or run.name == "nli-baseline"
        print("analyzing", run.name, flush=True)
        r = analyze_run(run, args.splits, args.seed, full)
        for suite, tasks in (("test_in", IN_DOMAIN), ("test_out", HELD_OUT)):
            if suite in r:
                r[suite]["macro_accuracy"] = macro(r[suite]["accuracy"], tasks)
                r[suite]["macro_accuracy_baseline_unseen"] = macro(r[suite]["accuracy"], tasks, BASELINE_SEEN)
        out["runs"][run.name] = r
    seeds = sorted(n for n in out["runs"] if n.startswith("decide-400m-s"))
    if seeds:
        agg = {}
        for suite in ("test_in", "test_out"):
            vals = {}
            for n in seeds:
                r = out["runs"][n][suite]
                for k, row in r["accuracy"]["by_kind"].items():
                    vals.setdefault(f"accuracy_{k}", []).append(row["accuracy"])
                for k, row in r["calibration"].items():
                    vals.setdefault(f"ece_raw_{k}", []).append(row["ece_raw"])
                    vals.setdefault(f"ece_calibrated_{k}", []).append(row["ece_calibrated"])
                for a in ("0.05", "0.10"):
                    for k, row in r["decide"][a].items():
                        vals.setdefault(f"coverage_{a}_{k}", []).append(row["coverage"])
                        vals.setdefault(f"error_{a}_{k}", []).append(row["error"])
                vals.setdefault("macro_accuracy", []).append(r["macro_accuracy"])
                vals.setdefault("macro_accuracy_baseline_unseen", []).append(r["macro_accuracy_baseline_unseen"])
                for t, rows in r["accuracy"]["by_task"].items():
                    for row in rows:
                        vals.setdefault(f"task_{t}_{row['tag']}", []).append(row["accuracy"])
            agg[suite] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                              "values": v} for k, v in vals.items()}
        out["seeds"] = {"runs": seeds, **agg}
    Path(args.out).write_text(json.dumps(out, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
