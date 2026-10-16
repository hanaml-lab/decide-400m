"""Download the public datasets, cast them as typed questions and write the training mixture and evaluation suites.

    python scripts/prepare_data.py --out data/mix

Writes, in the format of ``decide.data`` plus a ``task`` field and a ``tag`` per question:

* ``train.jsonl``: the training mixture (training tasks' train splits);
* ``dev.jsonl``: 300 states per training task from their train splits, not trained on, for the training log;
* ``test_in.jsonl``: the in-domain suite (training tasks' test or validation splits);
* ``test_out.jsonl``: the held-out suite (datasets no training example comes from);
* ``manifest.json``: each source's Hub id, commit, file and SHA-256, each task's licence and counts, and the
  SHA-256 of each output file. ``data/manifest.json`` in the repository is the copy from the reported run.

The files are written deterministically: the same sources give byte-identical outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from decide.tasks import CLINC, HELD_OUT_TASKS, SOURCES, TASKS, TRAIN_TASKS, cast_items, read_items, select

DEV = 300


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/mix")
    args = ap.parse_args(argv)
    from huggingface_hub import hf_hub_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}

    def fetch(source: str, split: str) -> str:
        repo, commit, names = SOURCES[source]
        path = hf_hub_download(repo, names[split], repo_type="dataset", revision=commit)
        files[f"{source}/{split}"] = {"repo": repo, "commit": commit, "file": names[split], "sha256": sha256(Path(path))}
        return path

    files["clinc/all"] = {"repo": "github.com/clinc/oos-eval (bundled)", "file": "data/clinc150.json.gz",
                          "sha256": sha256(CLINC)}
    train_rows, dev_rows, seen = [], [], set()
    stats: dict[str, dict] = {}
    for task in TRAIN_TASKS:
        t = TASKS[task]
        items = select(task, read_items(task, "train", fetch), 0)
        chosen, dev = items[: t.train_cap], items[t.train_cap: t.train_cap + DEV]
        seen.update(it.text for it in items[: t.train_cap + DEV])
        train_rows += list(cast_items(task, chosen, training=True))
        dev_rows += list(cast_items(task, dev, training=False))
        stats[task] = {"name": t.name, "kind": t.kind, "split": "train", "licence": t.licence,
                       "train_states": len(chosen), "available": len(items)}
    eval_in, eval_out = [], []
    for group, tasks, rows in (("in-domain", TRAIN_TASKS, eval_in), ("held-out", HELD_OUT_TASKS, eval_out)):
        for task in tasks:
            t = TASKS[task]
            items = [it for split in t.eval_splits for it in read_items(task, split, fetch)]
            fresh = [it for it in items if it.text not in seen]
            chosen = select(task, fresh, t.eval_cap)
            rows += list(cast_items(task, chosen, training=False))
            stats.setdefault(task, {"name": t.name, "kind": t.kind, "licence": t.licence})
            stats[task].update({"group": group, "eval_splits": list(t.eval_splits), "eval_states": len(chosen),
                                "dropped_as_seen_in_training": len(items) - len(fresh)})
    for name, rows in (("train", train_rows), ("dev", dev_rows), ("test_in", eval_in), ("test_out", eval_out)):
        with open(out / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        for task in {r["task"] for r in rows}:
            kinds = Counter((q["kind"], q["tag"]) for r in rows if r["task"] == task for q in r["questions"])
            none = sum(q["kind"] == "choice" and q["answer"] is None for r in rows if r["task"] == task for q in r["questions"])
            stats[task].setdefault("questions", {})[name] = {f"{k}/{tag}": n for (k, tag), n in sorted(kinds.items())}
            stats[task]["questions"][name]["choice_none_fits"] = none
    manifest = {"sources": files, "tasks": stats, "train_tasks": TRAIN_TASKS, "held_out_tasks": HELD_OUT_TASKS,
                "outputs": {f"{name}.jsonl": sha256(out / f"{name}.jsonl") for name in ("train", "dev", "test_in", "test_out")},
                "counts": {"train": len(train_rows), "dev": len(dev_rows), "test_in": len(eval_in), "test_out": len(eval_out)}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
