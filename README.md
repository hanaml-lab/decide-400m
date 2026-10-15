# decide-400m: typed answers about a state, with a bound on the error rate of the decisions it makes

[![ci](https://github.com/hanaml-lab/decide-400m/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/decide-400m/actions/workflows/ci.yml)

Reference implementation of [decide-400m](https://hanaml.com/models/decide-400m), hanaML's decision model. It reads an application's state (a support ticket, a log line, an agent's step, any serialised record) and a list of typed questions about it, and answers all of them in one encoder pass, with no generated text:

- **choice**: a distribution over the options given, plus a *none fits* slot;
- **yes/no**: a Bernoulli probability;
- **score**: a distribution over the levels of an ordered scale, from a cumulative-link head.

Every answer is calibrated, and every answer is either **decided** or **deferred**. A threshold set on calibration questions certifies that the error rate of the decisions made is at most the level α you choose, with probability 1 − δ over the calibration draw; answers below it go to a person or a larger model. That guarantee is the one the lab's [CCV](https://hanaml.com/research/papers/verifier-search-agents) uses to decide when a coding agent should submit, and [InForce](https://hanaml.com/research/papers/agentic-rag-citation-verification) to decide which claims an answer ships, applied here to every kind of answer.

## What is in it

- **Tree packing** (`decide/packing.py`, `decide/encoders.py`). The state and every question are laid out in one sequence, but attention follows a tree: the state sees only itself, a question sees the state and itself, an option sees the state, its question and itself. All questions start at the same rotary position, and so do all options of a question. Nothing in the encoder can tell which option or question came first: permuting options permutes their scores exactly, a question's answer does not change with the other questions asked beside it, and the state is encoded once for all of them. `TreeMaskedModernBert` runs a Hugging Face ModernBERT under this mask, with its local sliding-window layers windowed by position id.
- **Heads** (`decide/heads.py`). A set-attention layer over {none, option 1, …, option K}, with no positions, compares the options and gives the *none fits* slot a look at all of them. A Bernoulli head for yes/no. A proportional-odds head for scores whose cutpoints come from the levels' own vectors and are increasing by construction.
- **Calibration** (`decide/calibrate.py`). One temperature per kind of answer, fitted by negative log-likelihood on held-out questions; expected calibration error.
- **Decide or defer** (`decide/risk.py`). `decision_threshold` tests each threshold on a fixed grid with an exact binomial test of "the error rate of the decisions made exceeds α" at δ / (grid size), and takes the lowest threshold it certifies (Learn then Test with a Bonferroni family). CCV's rule bounds the same rate in expectation, up to O(n^−1/2); this one holds with probability 1 − δ and is more conservative for it. `state_threshold` gives InForce's answer-level guarantee instead: conformal risk control of the share of states with any wrong decision.
- **Data** (`decide/data.py`). Questions as JSON Lines, several per state; `examples/data/states.jsonl` shows the format.
- **Training** (`decide/train.py`, `scripts/train.py`, `configs/decide-400m.json`). The full run fine-tunes ModernBERT-large with the heads and writes the weights and a calibration file per error level.
- **The CPU toy** (`decide/toy.py`, `decide/demo.py`), below.
- **Latency** (`bench/latency.py`): measured, with random weights.
- **The model card's numbers** (`card/make_card.py`): a seeded simulation, illustrative.

## Install

```bash
pip install -e ".[dev]"        # numpy and torch; pytest and ruff
pip install -e ".[hf]"         # + transformers, for ModernBERT
```

## Quickstart (CPU)

```bash
python -m decide.demo          # train, calibrate and evaluate the toy: about a minute and a half on a laptop CPU
python -m decide.demo --alpha 0.05 0.10 0.20
pytest -q
```

## Toy run (real output)

The toy asks typed questions about requests from CLINC150 (Larson et al., 2019). Five of its domains are in scope (banking, credit cards, travel, work, auto and commute); a fifth of the requests come from outside them, from its out-of-scope set and its other domains. Every request gets three questions, packed together and answered in one pass: a choice among eight intents (for 15% of in-scope requests, eight intents of another domain, so none fits), a yes/no ("Does the user want: …?") and a three-level score (unrelated, same area, exact need). The model is a two-layer rotary encoder trained from scratch with the three heads. Calibration and test questions are drawn from one pool and split at random, so they are exchangeable. Output of `python -m decide.demo` on a CPU (Python 3.13):

```
CLINC150 toy: 9375 training states, 1875 calibration, 2813 test; 3 questions per state, one pass each; 0.40M parameters

kind      accuracy   ECE raw  ECE cal.      T
choice       0.883     0.033     0.023   1.35
yes/no       0.924     0.031     0.013   1.37
score        0.708     0.016     0.015   1.25

decide or defer at alpha = 0.05, delta = 0.10 (thresholds set on calibration, read on test)
kind     threshold  decided  error among decided
choice       0.885    45.5%                 2.2%
yes/no       0.905    78.4%                 2.6%
score        0.970    15.8%                 2.5%
choice questions no option fits (32% of test): 1.2% decided; options-only softmax, same rule: 1.4% decided, 42.7% coverage, 2.0% error

decide or defer at alpha = 0.10, delta = 0.10 (thresholds set on calibration, read on test)
kind     threshold  decided  error among decided
choice       0.590    58.3%                 6.8%
yes/no       0.500   100.0%                 7.6%
score        0.690    41.5%                 6.9%
choice questions no option fits (32% of test): 4.5% decided; options-only softmax, same rule: 7.5% decided, 55.4% coverage, 6.6% error

options-only softmax that always answers: decides 100%, error 38.5%, and answers every one of the 898 questions no option fits
options shuffled on every test choice: largest change in any probability 6.8e-06
(74 s on CPU, torch 2.12.0+cu126, 4 threads)
```

What it shows:

- **Calibration.** One temperature per kind lowers ECE on every kind (0.033 to 0.023 for choices, 0.031 to 0.013 for yes/no), without changing an answer.
- **The bound holds, conservatively.** At every level and for every kind, the error rate of the decisions made on test questions is below α: 2.2–2.6% at α = 0.05, 6.8–7.6% at α = 0.10. The binomial test certifies with probability 0.9, so it leaves room.
- **It defers where it should.** Of the choice questions no option fits, it decides 1.2% at α = 0.05. A softmax over the options that always answers decides all of them, and is wrong on 38.5% of all choice questions.
- **Scores are the hardest to certify.** At α = 0.05 it decides only 15.8% of scores.
- **The none-fits slot helps modestly.** Read as a softmax over the options alone, the same model under the same rule decides fewer questions (42.7% against 45.5% at α = 0.05) and more of those no option fits at α = 0.10 (7.5% against 4.5%). Most of the work is done by the threshold. A model trained without the slot is not compared here.
- **Order cannot matter.** With every test choice's options shuffled, no probability moves by more than float rounding (6.8e-06).

The toy checks that the method works end to end on a problem with known answers. It does not reproduce the model card's evaluation, which is illustrative. `card/toy_run.md` records the environment.

## Asking questions

```python
from decide import Calibration, Choice, Score, YesNo, ask

answers = ask(model, tokenizer, state, [
    Choice("Which team should take this ticket?", ["Billing", "Technical support", "Shipping"]),
    YesNo("Is the customer asking for money back?"),
    Score("How urgent is it?", ["whenever", "this week", "today", "right now"]),
], Calibration.load("calibration-alpha0.10.json"))
for answer in answers:
    print(answer)      # decided or deferred, the answer, its probability
```

`answer.probs` holds the calibrated distribution, `answer.none` a choice's mass on *none fits*, and `answer.decided` whether the answer cleared its kind's threshold.

## Latency (measured)

`bench/latency.py` times the architecture with random weights: the time of a forward pass does not depend on the values of the weights, and the trained ones are not published. One request is a 512-token state with four questions (two choices of eight options, a yes/no and a five-level score), 673 tokens tree-packed. The comparison reads the same request the way a zero-shot NLI classifier does, once per (state, question and option) pair: 22 sequences, 11,638 tokens, batched. Medians, from `card/latency.json`:

| Device | decide-400m, one pass | One pass per option |
|---|---|---|
| NVIDIA RTX 4070 Ti SUPER, bf16 | 22.6 ms | 164.8 ms |
| Intel Core i9-14900KF, 8 threads, fp32 | 5.4 s | 53.1 s |

The CPU row was measured on a machine shared with other jobs, so its absolute times are slow; the ratio is the point. The architecture has 401.9M parameters: 394.8M in ModernBERT-large with the five marker tokens, 7.1M in the heads.

## The model card's numbers (illustrative)

decide-400m's weights are not published and its full evaluation has not been run. The accuracy, calibration and decision rows of its card come from `card/make_card.py`, a seeded simulation whose parameters are set from public zero-shot results on the same datasets; it runs the card's real procedures (temperature scaling, ECE, the decide-or-defer test) on simulated scores. The held-out suite is built from Banking77, CLINC150, LEDGAR, TREC, AG News, BoolQ, ANLI, FEVER, SST-5, Yelp-5 and STS-B. On every push, CI reruns the simulation and checks its output against `card/results.sha256`.

## Layout

```
src/decide/   packing.py  encoders.py  heads.py  model.py  calibrate.py  risk.py  data.py  tokenizer.py  train.py  toy.py  demo.py
configs/      toy.json  decide-400m.json
data/         clinc150.json.gz (CLINC150, Larson et al., 2019)
examples/     data/states.jsonl
scripts/      train.py  run_toy.py
bench/        latency.py
card/         make_card.py  check_results.py  results.json  results.sha256  latency.json  toy_run.md
tests/        pytest suite, including the tree mask on a small random ModernBERT
```

## Citation

```bibtex
@misc{hanaml2026decide400m,
  title        = {decide-400m: Answering typed questions about a state in one pass, or deferring to a person},
  author       = {{hanaML}},
  year         = {2026},
  howpublished = {Model card},
  url          = {https://hanaml.com/models/decide-400m}
}
```

## License

Apache-2.0. See `LICENSE`.
