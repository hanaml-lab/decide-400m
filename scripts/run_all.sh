#!/usr/bin/env bash
# Every run the report uses, in the order they were run on one RTX 4070 Ti SUPER (16 GB, shared with another
# training job; each of these jobs caps itself at 6.2 GB). Training writes runs/<name>/ (weights, not committed);
# evaluation writes results/<name>/ (committed). The last step recomputes every reported number from results/.
set -euo pipefail
export PYTHONPATH=src
python scripts/prepare_data.py --out data/mix
train() {  # config, name, seed
  python scripts/train.py --config "$1" --out "runs/$2" --seed "$3"
  python scripts/evaluate.py --run "runs/$2" --out "results/$2"
}
train configs/decide-400m.json decide-400m-s0 0
python scripts/baseline_nli.py --out results/nli-baseline
train configs/decide-400m.json decide-400m-s1 1
train configs/decide-400m.json decide-400m-s2 2
train configs/ablations/noslot.json noslot 0
train configs/ablations/tree-base-30k.json tree-base-30k 0
train configs/ablations/flat-base-30k.json flat-base-30k 0
train configs/ablations/separate-base-30k.json separate-base-30k 0
python bench/latency_real.py --run runs/decide-400m-s0 --out results/latency_real.json
python scripts/mask_example.py --out results/mask_example.json
python scripts/analyze.py --results results --out results/summary.json
