"""The zero-shot NLI baseline: a ModernBERT-large entailment classifier read once per option.

    python scripts/baseline_nli.py --out results/nli-baseline

The baseline is MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (Apache-2.0), the same backbone fine-tuned on NLI and
on classification data recast as NLI. Each natural question of the evaluation suites is read the way such a
classifier is used: the state is the premise, and each option is put into a hypothesis written for the task; a
choice or a score is a softmax over the options' entailment logits, a yes/no the entailment probability of its
hypothesis. BoolQ is left out: its questions would need rewriting as statements. Derived questions are left out.

Its training mix includes several of the suite's datasets (Banking77, AG News, Yahoo Answers, Yelp, MultiNLI, and
Rotten Tomatoes, whose sentences SST-5 is built from), so on those it is not zero-shot; ``scripts/analyze.py``
reports them apart. Writes ``predictions_{dev,test_in,test_out}.jsonl.gz`` in the format of
``scripts/evaluate.py``, with a choice's slot 0 null (it has no *none fits* slot) and a score's ``form`` "softmax".
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import torch

from decide.data import read
from decide.trainer import machine

MODEL = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
REVISION = "a51e07b524299e309dd2b88d48b0cfa2bd9ec598"  # the Hub commit the reported run used
STS = {"unrelated": "The two sentences are unrelated.",
       "different but on the same topic": "The two sentences are different but on the same topic.",
       "somewhat related": "The two sentences are somewhat related.",
       "roughly equivalent": "The two sentences are roughly equivalent.",
       "mostly the same": "The two sentences mean mostly the same thing.",
       "the same meaning": "The two sentences have the same meaning."}
#: task: how the premise and the hypotheses are written. ``after`` splits the hypothesis out of the question text.
TEMPLATES = {
    "clinc": {"hyp": "The user wants {}."},
    "ag_news": {"hyp": "This article is about {}."},
    "dbpedia": {"hyp": "This text is about {}."},
    "yahoo": {"hyp": "This question is about {}."},
    "mnli": {"after": "that: "},
    "yelp": {"hyp": "The reviewer gave it {}."},
    "trec": {"hyp": "This question asks for {}."},
    "banking77": {"hyp": "The customer is asking about {}."},
    "newsgroups": {"hyp": "This post is about {}."},
    "rte": {"after": "that: "},
    "scitail": {"after": "claim: "},
    "mrpc": {"after": "sentence: "},
    "sst5": {"hyp": "The sentiment of this review is {}."},
    "stsb": {"after": "sentence: ", "sts": True},
    "tweet_sentiment": {"hyp": "The sentiment of this tweet is {}."},
}
SEEN_IN_BASELINE_TRAINING = ["banking77", "ag_news", "yahoo", "yelp", "mnli", "sst5"]


def pairs_for(rec, qi):
    """(premise, hypotheses) for one question, or None when the baseline does not answer it."""
    t = TEMPLATES.get(rec.task)
    q = rec.questions[qi]
    if t is None or (rec.tags and rec.tags[qi] != "natural"):
        return None
    if "after" in t:
        extra = q.text.split(t["after"], 1)[1]
        if t.get("sts"):
            premise = f'Sentence one: "{rec.state}" Sentence two: "{extra}"'
            return premise, [STS[o] for o in q.options]
        return rec.state, [extra]
    return rec.state, [t["hyp"].format(o) for o in q.options]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/nli-baseline")
    ap.add_argument("--data", default="data/mix")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--gpu-memory-gb", type=float, default=4.0)
    args = ap.parse_args(argv)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_memory_gb * 2**30 / total))
    revision = REVISION
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=revision, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    ent = model.config.label2id["entailment"]
    summary = {"model": MODEL, "revision": revision, "machine": machine(), "suites": {}}
    for suite in ("dev", "test_in", "test_out"):
        t0 = time.time()
        records = list(read(Path(args.data) / f"{suite}.jsonl"))
        jobs = []  # (record, question, kind, premise, hypotheses)
        for ri, rec in enumerate(records):
            for qi, q in enumerate(rec.questions):
                got = pairs_for(rec, qi)
                if got:
                    jobs.append((ri, qi, q.kind, *got))
        flat = [(j, h) for j, job in enumerate(jobs) for h in job[4]]
        enc = tok([jobs[j][3] for j, _ in flat], [h for _, h in flat], truncation="only_first", max_length=512)
        lengths = [len(x) for x in enc["input_ids"]]
        order = sorted(range(len(flat)), key=lambda i: lengths[i])
        scores = [None] * len(flat)
        i = 0
        with torch.inference_mode():
            while i < len(order):
                n = max(1, args.batch_tokens // lengths[order[min(i + 255, len(order) - 1)]])
                idx = order[i: i + min(n, 256)]
                batch = tok.pad({"input_ids": [enc["input_ids"][k] for k in idx],
                                 "attention_mask": [enc["attention_mask"][k] for k in idx]}, return_tensors="pt")
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.float()
                for k, row in zip(idx, logits.cpu()):
                    scores[k] = (row[ent].item(), (row[ent] - row[1 - ent]).item())
                i += len(idx)
        per_job: dict[int, list] = {}
        for (j, _), s in zip(flat, scores):
            per_job.setdefault(j, []).append(s)
        with gzip.open(out / f"predictions_{suite}.jsonl.gz", "wt") as f:
            for j, (ri, qi, kind, _, hyps) in enumerate(jobs):
                s = per_job[j]
                rec = records[ri]
                if kind == "yesno":
                    logits, form = [round(s[0][1], 3)], "logit"
                elif kind == "choice":
                    logits, form = [None] + [round(e, 3) for e, _ in s], "slots"
                else:
                    logits, form = [round(e, 3) for e, _ in s], "softmax"
                f.write(json.dumps({"r": ri, "q": qi, "task": rec.task, "kind": kind, "tag": rec.tags[qi],
                                    "target": rec.answers[qi], "form": form, "k": len(rec.questions[qi].options),
                                    "logits": logits}) + "\n")
        summary["suites"][suite] = {"questions": len(jobs), "pairs": len(flat), "seconds": time.time() - t0}
        print(suite, summary["suites"][suite], flush=True)
    summary["seen_in_baseline_training"] = SEEN_IN_BASELINE_TRAINING
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
