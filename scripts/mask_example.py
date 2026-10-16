"""The attention tree and position ids of one real request from the in-domain suite, for the report's Figure 1.

    python scripts/mask_example.py --out results/mask_example.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from decide.data import read
from decide.encoders import tree_mask
from decide.model import collate
from decide.packing import pack
from decide.tokenizer import HFTokenizer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/mix/test_in.jsonl")
    ap.add_argument("--out", default="results/mask_example.json")
    args = ap.parse_args(argv)
    tok = HFTokenizer()
    # The shortest CLINC150 request whose choice lists eight options, so every block is visible and small.
    pick = None
    for r in read(args.data):
        if r.task == "clinc" and len(r.questions) == 3 and len(r.questions[0].options) == 8:
            p = pack(tok, r.state, r.questions)
            if pick is None or len(p) < len(pick[1]):
                pick = (r, p)
    r, p = pick
    flat = pack(tok, r.state, r.questions, layout="flat")
    b = collate([p], tok.pad_id)
    mask = tree_mask(b.qgroup, b.ogroup)[0].to(torch.int8).tolist()
    tokens = [tok.tok.convert_ids_to_tokens(i) for i in p.ids]
    out = {"state": r.state, "questions": [{"kind": q.kind, "text": q.text, "options": list(q.options)} for q in r.questions],
           "tokens": tokens, "qgroup": list(p.qgroup), "ogroup": list(p.ogroup), "positions": list(p.positions),
           "flat_positions": list(flat.positions), "q_markers": p.q_markers, "opt_markers": p.opt_markers, "mask": mask}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out) + "\n")
    print(f"{len(p)} tokens; state: {r.state!r}")


if __name__ == "__main__":
    main()
