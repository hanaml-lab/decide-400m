"""Train and calibrate decide-400m from JSON Lines files (``decide.data``) on a GPU.

    python scripts/train.py --config configs/decide-400m.json --out runs/decide-400m

Writes the weights (``model.pt``) and ``calibration.json``: a temperature per kind and the decide-or-defer
threshold of each kind at each error level in the config, set on the calibration file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from decide.calibrate import Calibration, KINDS, fit_temperatures
from decide.data import read
from decide.encoders import TreeMaskedModernBert
from decide.model import DecisionModel, probabilities, verdicts
from decide.risk import decision_threshold
from decide.tokenizer import HFTokenizer
from decide.train import TrainConfig, fit, infer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/decide-400m.json")
    ap.add_argument("--out", default="runs/decide-400m")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = HFTokenizer(cfg["backbone"])
    encoder = TreeMaskedModernBert.from_pretrained(cfg["backbone"])
    encoder.resize(len(tok))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecisionModel(encoder, heads=16).to(device)
    train, cal = list(read(cfg["train"])), list(read(cfg["calibration"]))
    fit(model, tok, train, TrainConfig(cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["weight_decay"], cfg["warmup"], cfg["seed"]), log=print)
    torch.save(model.state_dict(), out / "model.pt")
    logits, targets = infer(model, tok, cal)
    temps = fit_temperatures(logits, targets)
    picks = verdicts(probabilities(logits, temps))
    for alpha in cfg["alphas"]:
        thresholds = {}
        for k in KINDS:
            if not len(targets[k]):
                continue
            pick, conf = picks[k]
            wrong = ((pick + 1 if k == "choice" else pick) != targets[k]).numpy()
            thresholds[k] = decision_threshold(conf.numpy(), wrong, alpha=alpha, delta=cfg["delta"]).value
        counts = {k: int(len(targets[k])) for k in KINDS}
        Calibration(temps, thresholds, alpha, cfg["delta"], counts).save(out / f"calibration-alpha{alpha:.2f}.json")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
