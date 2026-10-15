"""The CPU toy, end to end: train a small tree-packed decision model on CLINC150 questions, calibrate it, set the
decide-or-defer thresholds, and report what happens on held-out questions.

    python -m decide.demo                 # about a minute on a laptop CPU
    python -m decide.demo --alpha 0.10    # another error level
    python -m decide.demo --json          # the numbers as JSON
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .calibrate import KINDS, ece, fit_temperatures, top_label
from .encoders import TinyEncoder
from .model import DecisionModel, Logits, probabilities, verdicts
from .packing import Question
from .risk import decision_threshold, realised
from .tokenizer import WordTokenizer
from .toy import build
from .train import TrainConfig, fit, infer

NAMES = {"choice": "choice", "yesno": "yes/no", "score": "score"}
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "toy.json"
DEFAULTS = {"d": 64, "layers": 2, "heads": 4, "ff": 256, "epochs": 4, "batch_size": 64, "lr": 2e-3, "threads": 4,
            "alphas": [0.05, 0.10], "delta": 0.1}


def settings() -> dict:
    """The toy's settings: configs/toy.json where the checkout has it, else the same values built in."""
    if CONFIG.exists():
        return {**DEFAULTS, **{k: v for k, v in json.loads(CONFIG.read_text()).items() if not k.startswith("_")}}
    return dict(DEFAULTS)


def wrong_of(out: Logits, targets, temps) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each question's confidence and whether its answer, if decided, would be wrong."""
    picks = verdicts(probabilities(out, temps))
    result = {}
    for k in KINDS:
        pick, conf = picks[k]
        right = (pick + 1 if k == "choice" else pick) == targets[k]
        result[k] = (conf.numpy(), (~right).numpy())
    return result


def options_only(out: Logits, targets, cal_out: Logits, cal_targets):
    """The same model with its none-fits slot removed: a softmax over the given options only, recalibrated."""
    from .calibrate import fit_temperature

    def conf_wrong(o, t, temp):
        p = torch.softmax(o.choice[:, 1:] / temp, dim=-1)
        conf, pick = p.max(dim=-1)
        return conf.numpy(), (pick + 1 != t["choice"]).numpy()

    fits = cal_targets["choice"] > 0
    def nll(temp):
        logp = torch.log_softmax(cal_out.choice[fits, 1:] / temp, dim=-1)
        return -logp.gather(1, (cal_targets["choice"][fits] - 1)[:, None]).mean().item()
    temp = fit_temperature(nll)
    return conf_wrong(cal_out, cal_targets, temp), conf_wrong(out, targets, temp)


def shuffled(examples, seed: int):
    """The same examples with every choice's options in a new order, and targets moved to match."""
    rng = random.Random(seed)
    out = []
    for ex in examples:
        q = ex.questions[0]
        order = list(range(len(q.options)))
        rng.shuffle(order)
        new = Question("choice", q.text, tuple(q.options[i] for i in order))
        t = dict(ex.targets)
        if t["choice"]:
            t["choice"] = order.index(t["choice"] - 1) + 1
        out.append(type(ex)(ex.state, [new] + ex.questions[1:], t, ex.none_fits, ex.source))
    return out, order


def run(seed: int = 0, alphas=None, delta=None, epochs=None, threads=None, log=print) -> dict:
    cfg = settings()
    alphas = alphas or cfg["alphas"]
    delta = delta if delta is not None else cfg["delta"]
    epochs = epochs or cfg["epochs"]
    t0 = time.time()
    torch.set_num_threads(threads or cfg["threads"])
    torch.manual_seed(seed)
    train, cal, test = build(seed)
    texts = [ex.state for ex in train] + [q.text for ex in train for q in ex.questions]
    texts += [o for ex in train for q in ex.questions for o in q.options]
    tok = WordTokenizer.build(texts)
    model = DecisionModel(TinyEncoder(len(tok), d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], ff=cfg["ff"], dropout=0.0))
    params = sum(p.numel() for p in model.parameters())
    log(f"CLINC150 toy: {len(train)} training states, {len(cal)} calibration, {len(test)} test; "
        f"3 questions per state, one pass each; {params / 1e6:.2f}M parameters")
    history = fit(model, tok, train, TrainConfig(epochs=epochs, batch_size=cfg["batch_size"], lr=cfg["lr"], seed=seed))
    cal_out, cal_t = infer(model, tok, cal)
    out, t = infer(model, tok, test)
    temps = fit_temperatures(cal_out, cal_t)
    res: dict = {"seed": seed, "train_states": len(train), "calibration_states": len(cal), "test_states": len(test),
                 "parameters": params, "final_loss": history[-1], "kinds": {}, "alphas": {}}

    log("")
    log(f"{'kind':8s} {'accuracy':>9s} {'ECE raw':>9s} {'ECE cal.':>9s} {'T':>6s}")
    for k in KINDS:
        acc = float(top_label(out, t, k, temps[k])[1].mean())
        raw = ece(*top_label(out, t, k, 1.0))
        scaled = ece(*top_label(out, t, k, temps[k]))
        res["kinds"][k] = {"accuracy": acc, "ece_raw": raw, "ece_calibrated": scaled, "temperature": temps[k]}
        log(f"{NAMES[k]:8s} {acc:9.3f} {raw:9.3f} {scaled:9.3f} {temps[k]:6.2f}")

    cal_cw, test_cw = wrong_of(cal_out, cal_t, temps), wrong_of(out, t, temps)
    none = np.array([ex.none_fits for ex in test])
    (cal_oc, cal_ow), (test_oc, test_ow) = options_only(out, t, cal_out, cal_t)
    for alpha in alphas:
        log("")
        log(f"decide or defer at alpha = {alpha:.2f}, delta = {delta:.2f} (thresholds set on calibration, read on test)")
        log(f"{'kind':8s} {'threshold':>9s} {'decided':>8s} {'error among decided':>20s}")
        row = {}
        for k in KINDS:
            th = decision_threshold(*cal_cw[k], alpha=alpha, delta=delta)
            cov, err = realised(*test_cw[k], th.value)
            row[k] = {"threshold": th.value, "coverage": cov, "error": err}
            shown = f"{th.value:9.3f}" if th.value != float("inf") else f"{'none':>9s}"
            log(f"{NAMES[k]:8s} {shown} {cov:8.1%} {(f'{err:.1%}' if cov else 'defers all'):>20s}")
        th_c = row["choice"]["threshold"]
        decided_none = float((test_cw["choice"][0][none] >= th_c).mean())
        th_o = decision_threshold(cal_oc, cal_ow, alpha=alpha, delta=delta)
        cov_o, err_o = realised(test_oc, test_ow, th_o.value)
        decided_none_o = float((test_oc[none] >= th_o.value).mean())
        row["none_fits_decided"] = decided_none
        row["options_only"] = {"threshold": th_o.value, "coverage": cov_o, "error": err_o, "none_fits_decided": decided_none_o}
        res["alphas"][f"{alpha:.2f}"] = row
        log(f"choice questions no option fits ({none.mean():.0%} of test): {decided_none:.1%} decided; "
            f"options-only softmax, same rule: {decided_none_o:.1%} decided, {cov_o:.1%} coverage, {err_o:.1%} error")

    always_err = float(test_ow.mean())
    res["always_answer_error"] = always_err
    log("")
    log(f"options-only softmax that always answers: decides 100%, error {always_err:.1%}, "
        f"and answers every one of the {int(none.sum())} questions no option fits")

    moved, _ = shuffled(test, seed + 1)
    out2, _ = infer(model, tok, moved)
    p1 = torch.softmax(out.choice, -1)
    p2 = torch.softmax(out2.choice, -1)
    # Compare each option's probability across the two orders: none-fits slot and the right answer's slot.
    diffs = []
    for r, ex in enumerate(test):
        q1, q2 = ex.questions[0], moved[r].questions[0]
        a = {o: p1[r, i + 1].item() for i, o in enumerate(q1.options)}
        b = {o: p2[r, i + 1].item() for i, o in enumerate(q2.options)}
        diffs.append(max(abs(a[o] - b[o]) for o in a))
        diffs.append(abs(p1[r, 0].item() - p2[r, 0].item()))
    res["max_shuffle_change"] = max(diffs)
    res["seconds"] = time.time() - t0
    log(f"options shuffled on every test choice: largest change in any probability {max(diffs):.1e}")
    log(f"({time.time() - t0:.0f} s on CPU, torch {torch.__version__}, {torch.get_num_threads()} threads)")
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="decide-400m's method on the CPU toy (CLINC150).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, nargs="*", default=None, help="error levels (default: configs/toy.json)")
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.json:
        res = run(args.seed, args.alpha, args.delta, args.epochs, args.threads, log=lambda *_: None)
        print(json.dumps(res, indent=1))
        return res
    return run(args.seed, args.alpha, args.delta, args.epochs, args.threads)


if __name__ == "__main__":
    main()
