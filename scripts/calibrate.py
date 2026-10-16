"""Calibrate a trained decide-400m on your own labelled questions: temperatures and decide-or-defer thresholds.

    python scripts/calibrate.py --run runs/decide-400m-s0 --questions my_questions.jsonl --alpha 0.10 \
        --out calibration-alpha0.10.json

``--questions`` is JSON Lines in the format of ``decide.data``: states from your own traffic with their questions and
right answers. The temperatures are fitted on ``--temperatures-from`` (default: the run's dev states, as in the
report), so that the confidence is fixed before the thresholds see your questions; the thresholds are then set on
your questions by ``decide.risk.decision_threshold`` (Learn then Test) at ``--alpha`` and ``--delta``. The output loads
with ``decide.Calibration.load`` and is what ``decide.ask`` uses. The guarantee covers questions drawn like yours:
recalibrate when your traffic changes (the report's Section 5.5).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from decide.calibrate import KINDS, Calibration, fit_temperatures
from decide.data import read
from decide.model import probabilities, verdicts
from decide.risk import decision_threshold
from decide.tokenizer import HFTokenizer
from decide.trainer import CachedTokenizer, RunConfig, build_model, pack_record, predict, texts_of


def logits_of(model, records, cfg, device, tok):
    from decide.model import Logits

    rows = predict(model, [pack_record(tok, r, cfg) for r in records], records, cfg, device, tok.pad_id)
    by = {k: [r for r in rows if r["kind"] == k] for k in KINDS}
    width = max((len(r["logits"]) for r in by["choice"]), default=1)
    choice = torch.full((len(by["choice"]), width), float("-inf"))
    for i, r in enumerate(by["choice"]):
        choice[i, : len(r["logits"])] = torch.tensor(r["logits"])
    yesno = torch.tensor([r["logits"][0] for r in by["yesno"]])
    cw = max((len(r["logits"]) - 1 for r in by["score"]), default=1)
    eta = torch.tensor([r["logits"][0] for r in by["score"]])
    cuts = torch.full((len(by["score"]), cw), float("inf"))
    for i, r in enumerate(by["score"]):
        cuts[i, : len(r["logits"]) - 1] = torch.tensor(r["logits"][1:])
    targets = {k: torch.tensor([r["target"] for r in by[k]], dtype=torch.long) for k in KINDS}
    return Logits(choice, yesno, eta, cuts, {}), targets


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--temperatures-from", default="data/mix/dev.jsonl")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = RunConfig.load(Path(args.run) / "config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mine, temp_set = list(read(args.questions)), list(read(args.temperatures_from))
    tok = CachedTokenizer(HFTokenizer(cfg.backbone), texts_of(mine) + texts_of(temp_set), cfg.max_state)
    model = build_model(cfg, len(tok), device)
    model.load_state_dict(torch.load(Path(args.run) / "model.pt", map_location=device))
    t_logits, t_targets = logits_of(model, temp_set, cfg, device, tok)
    temps = fit_temperatures(t_logits, t_targets)
    logits, targets = logits_of(model, mine, cfg, device, tok)
    picks = verdicts(probabilities(logits, temps))
    thresholds, counts = {}, {}
    for k in KINDS:
        if not len(targets[k]):
            continue
        pick, conf = picks[k]
        wrong = ((pick + 1 if k == "choice" else pick) != targets[k]).numpy()
        th = decision_threshold(conf.numpy(), wrong, alpha=args.alpha, delta=args.delta)
        thresholds[k], counts[k] = th.value, int(len(targets[k]))
        print(f"{k}: threshold {th.value:.3f}, decides {th.coverage:.1%} of your {th.n} questions ({th.errors} wrong)")
    Calibration(temps, thresholds, args.alpha, args.delta, counts).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
