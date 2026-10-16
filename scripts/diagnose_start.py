"""Diagnostic: does one easy task (AG News choice questions) start to train, by layout, encoder learning rate,
optimizer and backbone? Output of the reported runs: results/diagnostics/start_sanity.txt (README, "Training").

    python scripts/diagnose_start.py tree marker 3e-5 adamw8bit answerdotai/ModernBERT-large
"""
import random
import sys
import time

import torch
from decide.trainer import RunConfig, CachedTokenizer, pack_record, make_batch, targets_for, build_model, texts_of, optimizer_for
from decide.data import read, Record
from decide.model import loss_terms
from decide.tokenizer import HFTokenizer
layout, readout, lr, optname = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
backbone = sys.argv[5] if len(sys.argv) > 5 else "answerdotai/ModernBERT-large"
cfg = RunConfig.load("configs/decide-400m.json", layout=layout, readout=readout, lr=lr, optimizer=optname, backbone=backbone)
recs = [r for r in read(cfg.train) if r.task == "ag_news"][:4000]
recs = [Record(r.state, r.questions[:1], r.answers[:1], r.task, r.tags[:1]) for r in recs]  # the choice only
tok = CachedTokenizer(HFTokenizer(cfg.backbone), texts_of(recs), cfg.max_state)
items = [pack_record(tok, r, cfg) for r in recs]
torch.manual_seed(0)
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(6.6 * 2**30 / total)
model = build_model(cfg, len(tok), "cuda")
if optname == "adamw":
    model.encoder.model.embeddings.tok_embeddings.weight.requires_grad_(False)
opt = optimizer_for(model, cfg)
rng = random.Random(0)
order = list(range(len(items))); rng.shuffle(order)
t0 = time.time(); run = []
for step in range(150):
    g = order[step * 16: (step + 1) * 16]
    model.train()
    b = make_batch(items, g, tok.pad_id).to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b)
    terms = loss_terms(out, targets_for(recs, g, out.where, "cuda"))
    loss = terms["choice"][0] / terms["choice"][1]
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for grp in opt.param_groups:
        grp["lr"] = (cfg.lr if grp["lr"] < 1e-4 or grp is not opt.param_groups[2] else cfg.head_lr) * min(1.0, (step + 1) / 20)
    opt.step()
    run.append(loss.item())
    if (step + 1) % 25 == 0:
        print(backbone.split("-")[-1], layout, readout, lr, optname, "step", step + 1, "loss", round(sum(run[-25:]) / 25, 3), f"{time.time() - t0:.0f}s", flush=True)
