"""Diagnostic: which parameters carry the gradient norm at initialisation (README, "Training").

    python scripts/diagnose_gradients.py
"""
import random
from collections import defaultdict

import torch
from decide.trainer import RunConfig, CachedTokenizer, pack_record, steps_of, make_batch, targets_for, build_model, texts_of
from decide.data import read
from decide.model import loss_terms
from decide.tokenizer import HFTokenizer
cfg = RunConfig.load("configs/decide-400m.json")
recs = list(read(cfg.train))[:4000]
tok = CachedTokenizer(HFTokenizer(cfg.backbone), texts_of(recs), cfg.max_state)
items = [pack_record(tok, r, cfg) for r in recs]
torch.manual_seed(0)
model = build_model(cfg, len(tok), "cuda"); model.train()
steps = steps_of(items, RunConfig(batch_size=32, token_budget=6144), random.Random(0))[:6]
for si, micro in enumerate(steps):
    nq = sum(len(recs[i].questions) for g in micro for i in g)
    for g in micro:
        b = make_batch(items, g, tok.pad_id).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        terms = loss_terms(out, targets_for(recs, g, out.where, "cuda"))
        (torch.stack([v for v, _ in terms.values()]).sum() / nq).backward()
    groups = defaultdict(float)
    for n, p in model.named_parameters():
        if p.grad is None: continue
        key = "heads." + n.split(".")[0] if not n.startswith("encoder") else ("embeddings" if "embeddings" in n else ("final_norm" if "final_norm" in n else "layer" + str(int(n.split(".")[3]) // 7)))
        groups[key] += p.grad.float().pow(2).sum().item()
    tot = sum(groups.values()) ** 0.5
    print(si, f"total {tot:.2f}", {k: round(v ** 0.5, 2) for k, v in sorted(groups.items())})
    emb = model.encoder.model.embeddings.tok_embeddings.weight.grad
    print("   marker rows grad norm", [round(x, 3) for x in emb[-5:].float().norm(dim=1).tolist()], "other rows", round(emb[:-5].float().norm().item(), 3))
    model.zero_grad()
