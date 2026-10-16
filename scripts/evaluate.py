"""Run a trained model over the evaluation suites and write every question's raw output.

    python scripts/evaluate.py --run runs/decide-400m-s0 --out results/decide-400m-s0

Writes to ``--out``:

* ``predictions_dev.jsonl.gz``, ``predictions_test_in.jsonl.gz`` and ``predictions_test_out.jsonl.gz``: one line per question with the record and
  question index, task, kind, tag, target and raw logits (a choice's 1 + K slot logits, slot 0 *none fits*; a
  yes/no's logit; a score's eta and cutpoints), rounded to 1e-3. ``scripts/analyze.py`` computes every reported
  number from these files.
* ``invariance.json``: the largest change in any option's probability, and the share of choice questions whose
  answer changes, when every choice's options are put in a new seeded order; and, for tree packing, the largest
  change in any output when each question is asked alone instead of packed with the state's other questions.
* ``eval_summary.json``: the machine, the time taken and the counts.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from pathlib import Path

import torch

from decide.data import Record, read
from decide.packing import Question
from decide.tokenizer import HFTokenizer
from decide.trainer import CachedTokenizer, RunConfig, build_model, machine, pack_record, predict, texts_of

SUITES = ("dev", "test_in", "test_out")  # dev: held-back training-task states, where temperatures are fitted


def write_predictions(path: Path, rows: list[dict], records: list[Record]) -> None:
    with gzip.open(path, "wt") as f:
        for r in rows:
            rec = records[r["record"]]
            q = rec.questions[r["question"]]
            f.write(json.dumps({"r": r["record"], "q": r["question"], "task": rec.task, "kind": r["kind"],
                                "tag": rec.tags[r["question"]] if rec.tags else "", "target": r["target"],
                                "form": {"choice": "slots", "yesno": "logit", "score": "cumlink"}[r["kind"]],
                                "k": len(q.options), "logits": [round(x, 3) for x in r["logits"]]}) + "\n")


def shuffled(records: list[Record], seed: int) -> tuple[list[Record], list[dict]]:
    """Every choice question with its options in a new order (never the same order); targets moved to match."""
    rng = random.Random(seed)
    out, perms = [], []
    for r in records:
        qs, ans, perm = [], [], {}
        for i, (q, a) in enumerate(zip(r.questions, r.answers)):
            if q.kind == "choice" and len(q.options) > 1:
                order = list(range(len(q.options)))
                while order == sorted(order):
                    rng.shuffle(order)
                qs.append(Question("choice", q.text, tuple(q.options[j] for j in order)))
                ans.append(order.index(a - 1) + 1 if a else 0)
                perm[i] = order
            else:
                qs.append(q)
                ans.append(a)
        out.append(Record(r.state, qs, ans, r.task, r.tags))
        perms.append(perm)
    return out, perms


def softmax(z):
    m = max(z)
    e = [pow(2.718281828459045, x - m) if x > -1e30 else 0.0 for x in z]
    s = sum(e)
    return [x / s for x in e]


def invariance(base: list[dict], moved: list[dict], perms: list[dict]) -> dict:
    """Compares each choice question's option probabilities across the two orders."""
    by = {(r["record"], r["question"]): r for r in moved}
    worst, flips, n, none_worst = 0.0, 0, 0, 0.0
    for r in base:
        if r["kind"] != "choice" or r["question"] not in perms[r["record"]]:
            continue
        order = perms[r["record"]][r["question"]]
        p = softmax(r["logits"])
        q = softmax(by[(r["record"], r["question"])]["logits"])
        # option j of the original is option order.index(j) of the shuffled one
        diffs = [abs(p[1 + j] - q[1 + order.index(j)]) for j in range(len(order))]
        worst = max(worst, max(diffs))
        none_worst = max(none_worst, abs(p[0] - q[0]))
        a = max(range(len(order)), key=lambda j: p[1 + j])
        b = order[max(range(len(order)), key=lambda j: q[1 + j])]
        flips += int(a != b)
        n += 1
    return {"choice_questions": n, "max_option_prob_change": worst, "max_none_prob_change": none_worst,
            "answer_changed": flips, "answer_changed_share": flips / max(n, 1)}


def alone(records: list[Record]) -> tuple[list[Record], list[tuple[int, int]]]:
    """Each question of a multi-question state as a state of its own."""
    out, where = [], []
    for ri, r in enumerate(records):
        if len(r.questions) < 2:
            continue
        for i, (q, a) in enumerate(zip(r.questions, r.answers)):
            out.append(Record(r.state, [q], [a], r.task, r.tags[i: i + 1]))
            where.append((ri, i))
    return out, where


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/mix")
    ap.add_argument("--limit", type=int, default=0, help="only the first n states of each suite (smoke tests)")
    ap.add_argument("--fp32-states", type=int, default=500, help="states with several questions for the fp32 checks")
    args = ap.parse_args(argv)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig.load(run / "config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and cfg.gpu_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg.gpu_memory_gb * 2**30 / total))
    t0 = time.time()
    suites = {s: list(read(Path(args.data) / f"{s}.jsonl")) for s in SUITES}
    if args.limit:
        suites = {s: v[: args.limit] for s, v in suites.items()}
    tok = CachedTokenizer(HFTokenizer(cfg.backbone), [t for v in suites.values() for t in texts_of(v)], cfg.max_state)
    model = build_model(cfg, len(tok), device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))
    model.eval()
    summary = {"run": str(run), "config": cfg.name, "layout": cfg.layout, "none_slot": cfg.none_slot,
               "seed": cfg.seed, "machine": machine(), "suites": {}}
    inv = {}
    for name, records in suites.items():
        s0 = time.time()
        items = [pack_record(tok, r, cfg) for r in records]
        rows = predict(model, items, records, cfg, device, tok.pad_id)
        write_predictions(out / f"predictions_{name}.jsonl.gz", rows, records)
        summary["suites"][name] = {"states": len(records), "questions": len(rows), "seconds": time.time() - s0}
        if name == "dev":
            continue
        moved, perms = shuffled(records, seed=1)
        tok.add(texts_of(moved))
        moved_rows = predict(model, [pack_record(tok, r, cfg) for r in moved], moved, cfg, device, tok.pad_id)
        inv[name] = {"option_order": invariance(rows, moved_rows, perms)}
        if cfg.layout == "tree":
            solo, where = alone(records)
            solo_rows = predict(model, [pack_record(tok, r, cfg) for r in solo], solo, cfg, device, tok.pad_id)
            ref = {(r["record"], r["question"]): r["logits"] for r in rows}
            worst = max((max(abs(a - b) for a, b in zip(ref[where[s["record"]]], s["logits"])) for s in solo_rows),
                        default=0.0)
            inv[name]["asked_alone"] = {"questions": len(solo_rows), "max_logit_change": worst}
        print(name, summary["suites"][name], inv.get(name), flush=True)
    # The same checks in fp32 on a subset: bf16 kernels round differently for differently padded shapes, which is
    # arithmetic noise, not information flowing between options or questions.
    cfg32 = RunConfig.load(run / "config.json", bf16=False)
    sub = [r for r in suites["test_out"] + suites["test_in"] if len(r.questions) > 1][: args.fp32_states]
    rows = predict(model, [pack_record(tok, r, cfg32) for r in sub], sub, cfg32, device, tok.pad_id)
    moved, perms = shuffled(sub, seed=1)
    moved_rows = predict(model, [pack_record(tok, r, cfg32) for r in moved], moved, cfg32, device, tok.pad_id)
    inv["fp32_subset"] = {"states": len(sub), "option_order": invariance(rows, moved_rows, perms)}
    if cfg.layout == "tree":
        solo, where = alone(sub)
        solo_rows = predict(model, [pack_record(tok, r, cfg32) for r in solo], solo, cfg32, device, tok.pad_id)
        ref = {(r["record"], r["question"]): r["logits"] for r in rows}
        inv["fp32_subset"]["asked_alone"] = {
            "questions": len(solo_rows),
            "max_logit_change": max(max(abs(a - b) for a, b in zip(ref[where[x["record"]]], x["logits"])) for x in solo_rows)}
    print("fp32 subset", inv["fp32_subset"], flush=True)
    summary["seconds"] = time.time() - t0
    (out / "invariance.json").write_text(json.dumps(inv, indent=1) + "\n")
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for f in ("config.json", "train_summary.json", "log.jsonl"):
        if (run / f).exists():
            (out / f).write_text((run / f).read_text())


if __name__ == "__main__":
    main()
