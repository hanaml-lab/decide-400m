"""Train decide-400m (or an ablation) on one GPU from a run config.

    python scripts/prepare_data.py --out data/mix          # once: the training mixture and the suites
    python scripts/train.py --config configs/decide-400m.json --out runs/decide-400m-s0 --seed 0

Writes to ``--out``: ``config.json`` (the resolved settings), ``log.jsonl`` (the training log), ``train_summary.json``
and ``model.pt`` (fp32 weights, about 1.6 GB; not committed). ``scripts/evaluate.py`` then writes the predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from decide.trainer import RunConfig, train


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/decide-400m.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    args = ap.parse_args(argv)
    cfg = RunConfig.load(args.config, seed=args.seed, name=args.name, max_steps=args.max_steps)
    train(cfg, Path(args.out))


if __name__ == "__main__":
    main()
