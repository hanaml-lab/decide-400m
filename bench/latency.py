"""Latency of decide-400m's architecture, measured with random weights (the time of a forward pass does not depend
on the values of the weights, and the trained weights are not published).

One request is a state of 512 tokens and four questions: two choices of eight options, a yes/no and a five-level
score. decide-400m reads it in one tree-packed pass. The comparison reads it the way a zero-shot NLI classifier
does: one pass over (state, question + option) per option, 22 sequences of the same encoder, batched.

    python bench/latency.py --device cpu --threads 8
    python bench/latency.py --device cuda
    python bench/latency.py --write results/latency_synthetic.json    # both, with the machine's description
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from decide.encoders import TreeMaskedModernBert
from decide.model import DecisionModel, collate
from decide.packing import Choice, Score, YesNo, pack

#: answerdotai/ModernBERT-large's configuration (config.json on the Hub), so no download is needed.
MODERNBERT_LARGE = dict(vocab_size=50368, hidden_size=1024, intermediate_size=2624, num_hidden_layers=28,
                        num_attention_heads=16, global_attn_every_n_layers=3, local_attention=128,
                        global_rope_theta=160000.0, local_rope_theta=10000.0, max_position_embeddings=8192,
                        pad_token_id=50283, cls_token_id=50281, sep_token_id=50282, bos_token_id=50281,
                        eos_token_id=50282, norm_bias=False, attention_bias=False, mlp_bias=False)
MARKERS = 5  # [Q], [OPT], [CHOICE], [YESNO], [SCORE]


class SyntheticTokenizer:
    """Fixed-length token streams: latency depends on lengths, not on which tokens they are."""

    cls_id, pad_id = 50281, 50283
    q_id, opt_id = 50368, 50369
    kind_ids = {"choice": 50370, "yesno": 50371, "score": 50372}

    def encode(self, text: str) -> list[int]:
        n = int(text.split(":")[0])
        return [1000 + i % 40000 for i in range(n)]


def request():
    state = "511:"  # 511 tokens after [CLS]
    questions = [Choice("12:", [f"{4 + i % 3}:{i}" for i in range(8)]), Choice("10:", [f"{3 + i % 4}:{i}" for i in range(8)]),
                 YesNo("14:"), Score("11:", [f"2:{i}" for i in range(5)])]
    return state, questions


def nli_pairs(tok, state, questions):
    """The token count of each (state, question + option) pair an NLI classifier would read."""
    n_state = 1 + len(tok.encode(state)) + 1
    pairs = []
    for q in questions:
        for option in (q.options or ("",)):
            pairs.append(n_state + len(tok.encode(q.text)) + (len(tok.encode(option)) if option else 0) + 1)
    return pairs


def timed(fn, device, runs):
    for _ in range(3 if device == "cuda" else 1):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    out = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out)


def measure(device: str, threads: int, runs: int) -> dict:
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(0)
    if device == "cpu":
        torch.set_num_threads(threads)
    config = ModernBertConfig(**MODERNBERT_LARGE, attn_implementation="sdpa")
    bert = ModernBertModel(config)
    bert.resize_token_embeddings(config.vocab_size + MARKERS)
    model = DecisionModel(TreeMaskedModernBert(bert), heads=16).eval()
    params = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for p in bert.parameters())
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype)
    tok = SyntheticTokenizer()
    state, questions = request()
    batch = collate([pack(tok, state, questions)], tok.pad_id)
    for name in ("ids", "positions", "qgroup", "ogroup"):
        setattr(batch, name, getattr(batch, name).to(device))
    lengths = nli_pairs(tok, state, questions)
    pairs = torch.full((len(lengths), max(lengths)), 1000, device=device)
    pad_mask = torch.arange(max(lengths), device=device)[None] < torch.tensor(lengths, device=device)[:, None]
    with torch.inference_mode():
        tree_ms = timed(lambda: model(batch), device, runs)
        nli_ms = timed(lambda: bert(input_ids=pairs, attention_mask=pad_mask), device, runs)
    return {"device": device, "dtype": str(dtype).replace("torch.", ""), "threads": threads if device == "cpu" else None,
            "tree_tokens": len(batch.ids[0]), "nli_sequences": len(lengths), "nli_tokens": sum(lengths),
            "tree_ms_p50": round(tree_ms, 1), "nli_ms_p50": round(nli_ms, 1), "runs": runs,
            "parameters": params, "backbone_parameters": backbone}


def machine() -> dict:
    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"cpu": cpu, "gpu": gpu, "torch": torch.__version__, "python": platform.python_version()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--runs", type=int, default=7, help="timed runs on the CPU; the GPU gets ten times as many")
    ap.add_argument("--write", default=None, help="measure on the CPU and, if present, the GPU, and write JSON here")
    args = ap.parse_args(argv)
    if args.write:
        rows = [measure("cpu", args.threads, args.runs)]
        if torch.cuda.is_available():
            rows.append(measure("cuda", args.threads, args.runs * 10))
        Path(args.write).write_text(json.dumps({"machine": machine(), "results": rows}, indent=1) + "\n")
        print(json.dumps(rows, indent=1))
        return rows
    row = measure(args.device or ("cuda" if torch.cuda.is_available() else "cpu"), args.threads, args.runs)
    print(json.dumps(row, indent=1))
    return row


if __name__ == "__main__":
    main()
