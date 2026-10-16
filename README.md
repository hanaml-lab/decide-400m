# decide-400m: typed answers about a state in one encoder pass, calibrated, or deferred under a certified error rate

[![ci](https://github.com/hanaml-lab/decide-400m/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/decide-400m/actions/workflows/ci.yml)

Reference implementation, training and evaluation code of [decide-400m](https://hanaml.com/models/decide-400m), hanaML's decision model, and of its technical report, [*Decide or Defer*](https://hanaml.com/research/papers/decide-400m-report). It reads an application's state (a support ticket, a log line, an agent's step, any serialised record) and a list of typed questions about it, and answers all of them in one encoder pass, with no generated text:

- **choice**: a distribution over the options given, plus a *none fits* slot;
- **yes/no**: a Bernoulli probability;
- **score**: a distribution over the levels of an ordered scale, from a cumulative-link head.

Every answer is calibrated, and every answer is either **decided** or **deferred**. A threshold set on labelled calibration questions certifies that the error rate of the decided answers is at most the level α you choose, with probability 1 − δ over the calibration draw (Learn then Test); the rest go to a person or a larger model. That is the guarantee the lab's [CCV](https://hanaml.com/research/papers/verifier-search-agents) uses to decide when a coding agent should submit, and [InForce](https://hanaml.com/research/papers/agentic-rag-citation-verification) to decide which claims an answer ships, applied to every kind of answer.

**v0.2.0** trains the model and measures it: ModernBERT-large fine-tuned on seven public datasets cast as typed questions, three seeds on one GPU, evaluated on the test splits of the training datasets and on nine datasets held out of training. Every number below and in the report is computed by `scripts/analyze.py` from the predictions committed in `results/`, and CI recomputes it on every push. The weights are not published.

## Measured results

Means over three seeds (0, 1, 2); decide-or-defer over 200 random calibration/test splits of each suite, δ = 0.1. The NLI baseline is [ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0), one pass per option; it trained on Banking77, AG News, Yahoo Answers, Yelp and MultiNLI itself, and on the source sentences of SST-5.

| | In-domain suite (7 datasets) | Held-out suite (9 datasets) |
| --- | --- | --- |
| Accuracy: choice / yes/no / score (%) | 89.3 / 92.9 / 75.4 | 63.9 / 76.7 / 43.9 |
| Macro accuracy, questions both answer: decide-400m / NLI baseline (%) | 86.8 / 77.6 | 59.9 / 66.5 |
| ECE, no temperature: choice / yes/no / score | 0.015 / 0.009 / 0.031 | 0.171 / 0.112 / 0.102 |
| Decided at α = 0.10, thresholds set on the same suite (%) | 84.8 (error 7.6) | 12.7 (error 6.3) |
| Decided at α = 0.05 (%) | 73.5 (error 4.0) | 0.7 (error 0.7) |

What the runs show:

- **In domain the rule decides most questions at a certified error rate.** Over 200 splits, no kind's test error exceeded α in more than 2% of them at any level, against the δ = 10% the guarantee allows.
- **The bound does not travel.** Thresholds set on the in-domain suite and applied to the held-out one let through errors of 20.9% (choice), 23.3% (yes/no) and 19.3% (score) at α = 0.10. Set on labelled held-out questions, they hold, but decide little until there are thousands of labels: calibrate on your own traffic with `scripts/calibrate.py`.
- **On new tasks the NLI baseline is more accurate.** It is ahead on every held-out dataset, by 0.8 to 13.4 points; it was trained on far more data.
- **None fits.** The slot separates questions no option fits from the rest with AUROC 0.990 in domain and 0.726 on held-out tasks.
- **Order cannot matter.** In fp32, shuffling every choice's options changed no answer (largest change in any probability 4.0 × 10<sup>−6</sup>).
- **Layouts.** At a matched budget on ModernBERT-base, an ordinary flat pass was about as accurate as tree packing but changed 13.4% of choice answers when the options were reordered (tree packing: none, in fp32). One pass per option did not start in any of four attempts.
- **Latency.** On real requests, one tree-packed pass took a median 908 ms on 8 CPU threads (fp32, beside another job) against 2,279 ms for one pass per option, 2.8 times faster per request. On an RTX 4070 Ti SUPER (bf16), one request at a time, it took 50.1 ms against 53.6 ms: there time barely depends on the tokens read. On the questions the NLI baseline can answer, 50.8 ms against its 64.5 ms.

## Reproduce

On one GPU with about 6 GB free (each job caps itself at 6.2 GB) and Python 3.11+:

```bash
pip install torch            # a CUDA build
pip install -e ".[train,analysis,dev]"
bash scripts/run_all.sh      # every reported run, in order
```

`scripts/run_all.sh` runs, in order: `prepare_data.py` (downloads the datasets at pinned Hub commits and writes the mixture and the suites, byte-identical to `data/manifest.json`'s digests); `train.py` for the three seeds and the four ablations (`configs/`); `evaluate.py` after each; `baseline_nli.py`; `bench/latency_real.py`; and `analyze.py`, which writes `results/summary.json`. One epoch took 44 minutes per seed on an RTX 4070 Ti SUPER shared with another job. The latency numbers are from a second run of `bench/latency_real.py` (default seed), made when no other compute job used the GPU (`results/diagnostics/latency_real_gpu_use.log`); the first, made while another job trained on it, is `results/diagnostics/latency_real_gpu_shared.json`. To check the numbers without a GPU:

```bash
pip install -e ".[analysis]"
python scripts/analyze.py --results results --out /tmp/summary.json
python scripts/check_summary.py /tmp/summary.json results/summary.json
```

## Training

`src/decide/trainer.py`: one epoch of steps of 32 states, length-grouped into micro-batches of at most 8,192 padded tokens; AdamW in its 8-bit form (bitsandbytes), learning rates 3e-5 (encoder) and 1e-4 (heads), 6% warm-up then linear decay; bf16 autocast with fp32 master weights and layer recomputation; peak 5.60 GB.

Two parts of the recipe come from failures, and the record of them is in `results/diagnostics/`:

- **The heads read standardised vectors.** With the heads reading ModernBERT's raw marker vectors, six starts on the full mixture sat at chance for 200 to 550 steps. A linear probe reads AG News topics from the frozen `[Q]` vector 75% of the time, but a fresh choice head on the raw vectors learns nothing, because they share a large common component (mean cosine 0.763 across states); on the same vectors standardised per feature it learns (`diagnose_features.py`, `diagnose_heads.py`). Each feature is now standardised with statistics from 2,000 training states under the untrained encoder.
- **A run that has not started by step 500 starts again.** Even with standardised inputs, leaving the initial plateau was partly chance: two starts with the same seed differed, one leaving it by step 200, the other not by step 350. The rule, fixed before any reported run: a run whose dev choice accuracy at step 500 is below 0.40 restarts with the same seed, up to four attempts. Of the reported runs, seed 0, seed 1, seed 2, the flat layout, the no-slot ablation and the tree layout on the base encoder started on the first attempt; the one-pass-per-option layout did not start in any of its four attempts.

A run trained with an error in MultiNLI's three-way answers (neutral and contradiction swapped) was stopped and is not reported; its log was not kept. A layer-wise learning-rate decay of 0.9 started training but learned slowly, and that run was stopped too (`results/diagnostics/full-llrd0.9-seed0-stopped-at-1500.jsonl`).

## What is in it

- `src/decide/packing.py`: the tree-packed layout, plus a flat layout and one pass per option for the ablations.
- `src/decide/encoders.py`: the tree mask; ModernBERT run under it with boolean masks, its sliding-window layers windowed by position id; a small encoder for the CPU toy.
- `src/decide/heads.py`, `model.py`: the choice head with its *none fits* slot (optional), the yes/no and cumulative-link score heads, standardised inputs, and `ask`.
- `src/decide/calibrate.py`, `risk.py`: per-kind temperatures, ECE, and the decide-or-defer threshold (Learn then Test, binomial tests at δ / 100 on a fixed grid).
- `src/decide/tasks.py`, `scripts/prepare_data.py`: public datasets cast as typed questions.
- `src/decide/trainer.py`, `scripts/train.py`, `configs/`: training on one GPU.
- `scripts/evaluate.py`, `scripts/baseline_nli.py`, `bench/latency_real.py`: predictions, the baseline and latency.
- `scripts/analyze.py`, `scripts/check_summary.py`: every reported number, from the predictions.
- `scripts/calibrate.py`: temperatures and thresholds for your own labelled questions.
- `src/decide/toy.py`, `demo.py`: the CPU toy below.

## Quickstart (CPU toy)

```bash
pip install -e ".[dev]"
python -m decide.demo          # trains, calibrates and evaluates a tiny model on CLINC150 in about a minute and a half
pytest -q
```

The toy and its real output are unchanged from v0.1.0: `card/toy_run.md`.

## Asking questions

```python
import torch
from decide import Calibration, Choice, Score, YesNo, ask
from decide.tokenizer import HFTokenizer
from decide.trainer import RunConfig, build_model

cfg = RunConfig.load("runs/decide-400m-s0/config.json")
tok = HFTokenizer(cfg.backbone)
model = build_model(cfg, len(tok), "cuda")
model.load_state_dict(torch.load("runs/decide-400m-s0/model.pt"))
# python scripts/calibrate.py --run runs/decide-400m-s0 --questions mine.jsonl --alpha 0.10 --out calibration.json
calibration = Calibration.load("calibration.json")

answers = ask(model, tok, open("ticket.txt").read(), [
    Choice("Which team should take this ticket?", ["Billing", "Technical support", "Shipping"]),
    YesNo("Is the customer asking for money back?"),
    Score("How urgent is it?", ["whenever", "this week", "today", "right now"]),
], calibration)
for answer in answers:
    print(answer)      # decided or deferred, the answer, its probability
```

## Citation

```bibtex
@techreport{agarwal2026decide,
  title       = {Decide or Defer: Typed Questions about Application State in One Encoder Pass, with a Certified Error Rate on the Decisions Made},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2026},
  url         = {https://hanaml.com/research/papers/decide-400m-report}
}
```

## License

Apache-2.0. See `LICENSE`. The datasets keep their own terms (`data/manifest.json` records what each Hub card states); two state none and Yelp's has its own agreement, which is one reason the weights are not published.
