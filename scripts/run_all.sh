#!/usr/bin/env bash
# Every run the report uses, in the order they were run on one RTX 4070 Ti SUPER (16 GB, shared; each job capped
# at about 6 GB). Training writes runs/<name>/ (weights not committed); evaluation writes results/<name>/.
set -euo pipefail
export PYTHONPATH=src
python scripts/prepare_data.py --out data/mix
for seed in 0 1 2; do
  python scripts/train.py --config configs/decide-400m.json --out runs/decide-400m-s$seed --seed $seed
  python scripts/evaluate.py --run runs/decide-400m-s$seed --out results/decide-400m-s$seed
done
python scripts/baseline_nli.py --out results/nli-baseline
for name in noslot tree-30k flat-30k separate-30k; do
  python scripts/train.py --config configs/ablations/$name.json --out runs/$name --seed 0
  python scripts/evaluate.py --run runs/$name --out results/$name
done
python bench/latency_real.py --run runs/decide-400m-s0 --out results/latency_real.json
python bench/latency.py --write results/latency_synthetic.json
python scripts/analyze.py --results results --out results/summary.json
