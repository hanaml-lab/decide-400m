"""Latency of the trained decide-400m on real requests from the evaluation suites, against reading the same
questions one pass per option with the same encoder, and against the zero-shot NLI baseline.

    python bench/latency_real.py --run runs/decide-400m-s0 --out results/latency_real.json

Requests are sampled from both suites, the same number from each task (seeded), each with all the questions the
suite asks of its state. Per request, batch size 1:

* ``tree``: decide-400m, one tree-packed pass;
* ``per_option``: the same encoder and heads reading the request one pass per option (``pack_separate``), all of a
  request's passes in one batch; this times the layout, not a trained model of it;
* ``nli``: MoritzLaurer/ModernBERT-large-zeroshot-v2.0 reading one (state, hypothesis) pair per option of the
  request's natural questions, all pairs in one batch, as a zero-shot classifier is used.

GPU in bf16, CPU in fp32 with ``--threads`` threads. Medians and 90th percentiles over requests, after warm-up.
The machine was shared with other jobs, so absolute CPU times are pessimistic; the ratios are the point.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline_nli import MODEL as NLI, REVISION as NLI_REVISION, pairs_for  # noqa: E402

from decide.data import read  # noqa: E402
from decide.model import collate, collate_separate  # noqa: E402
from decide.packing import pack, pack_separate  # noqa: E402
from decide.tokenizer import HFTokenizer  # noqa: E402
from decide.trainer import RunConfig, build_model, machine  # noqa: E402


def sample(data: Path, per_task: int, seed: int):
    rng = random.Random(seed)
    out = []
    for suite in ("test_in", "test_out"):
        records = list(read(data / f"{suite}.jsonl"))
        by = {}
        for r in records:
            by.setdefault(r.task, []).append(r)
        for task in sorted(by):
            out += [(suite, r) for r in rng.sample(by[task], min(per_task, len(by[task])))]
    return out


def timed(fn, device):
    for _ in range(2 if device == "cuda" else 1):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000


def measure(device: str, run: Path, requests, threads: int) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device == "cpu":
        torch.set_num_threads(threads)
    cfg = RunConfig.load(run / "config.json", checkpointing=False)
    tok = HFTokenizer(cfg.backbone)
    model = build_model(cfg, len(tok), "cpu")
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    ntok = AutoTokenizer.from_pretrained(NLI, revision=NLI_REVISION)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI, revision=NLI_REVISION, attn_implementation="sdpa",
                                                             torch_dtype=dtype).to(device).eval()
    rows = []
    with torch.inference_mode():
        for suite, r in requests:
            p = pack(tok, r.state, r.questions, max_state=cfg.max_state)
            tree = collate([p], tok.pad_id).to(device)
            s = pack_separate(tok, r.state, r.questions, max_state=cfg.max_state)
            sep = collate_separate([s], tok.pad_id).to(device)
            pairs = [pairs_for(r, i) for i in range(len(r.questions))]
            pairs = [(pr, h) for x in pairs if x for pr, hs in [x] for h in hs]
            row = {"suite": suite, "task": r.task, "questions": len(r.questions), "tree_tokens": len(p),
                   "per_option_passes": len(s.sequences), "per_option_tokens": sum(len(x) for x in s.sequences)}
            row["tree_ms"] = timed(lambda: model(tree), device)
            row["per_option_ms"] = timed(lambda: model(sep), device)
            if pairs:
                enc = ntok([a for a, _ in pairs], [b for _, b in pairs], truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt").to(device)
                row["nli_pairs"] = len(pairs)
                row["nli_tokens"] = int(enc["attention_mask"].sum())
                row["nli_ms"] = timed(lambda: nli(**enc), device)
                # decide-400m on the same natural questions only, for a like-for-like comparison
                nat = [q for q, tag in zip(r.questions, r.tags) if tag == "natural"]
                pn = collate([pack(tok, r.state, nat, max_state=cfg.max_state)], tok.pad_id).to(device)
                row["tree_natural_ms"] = timed(lambda: model(pn), device)
            rows.append(row)
    def stats(key, sel=lambda x: True):
        v = [x[key] for x in rows if key in x and sel(x)]
        return {"median": statistics.median(v), "p90": sorted(v)[int(0.9 * (len(v) - 1))], "n": len(v)} if v else None
    summary = {k: stats(k) for k in ("tree_ms", "per_option_ms", "nli_ms", "tree_natural_ms", "tree_tokens",
                                     "per_option_tokens", "nli_tokens")}
    multi = lambda x: x["questions"] >= 2  # noqa: E731
    summary["multi_question"] = {k: stats(k, multi) for k in ("tree_ms", "per_option_ms", "nli_ms")}
    return {"device": device, "dtype": str(dtype).replace("torch.", ""), "threads": threads if device == "cpu" else None,
            "summary": summary, "requests": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/mix")
    ap.add_argument("--per-task", type=int, default=25, help="requests per task on the GPU")
    ap.add_argument("--cpu-per-task", type=int, default=4, help="requests per task on the CPU")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-gb", type=float, default=4.0)
    args = ap.parse_args(argv)
    res = {"machine": machine(), "nli_model": NLI, "nli_revision": NLI_REVISION, "results": []}
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_memory_gb * 2**30 / total))
        res["results"].append(measure("cuda", Path(args.run), sample(Path(args.data), args.per_task, args.seed), args.threads))
        print(json.dumps(res["results"][-1]["summary"], indent=1), flush=True)
    res["results"].append(measure("cpu", Path(args.run), sample(Path(args.data), args.cpu_per_task, args.seed + 1), args.threads))
    print(json.dumps(res["results"][-1]["summary"], indent=1), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
