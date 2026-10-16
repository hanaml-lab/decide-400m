"""Diagnostic: can the choice head learn from frozen ModernBERT features? Caches the [Q] and [OPT] vectors of
AG News choice questions once, then trains a fresh ChoiceHead on them for 400 Adam steps: on the raw vectors, and
on vectors standardised with statistics from the training states.

    python scripts/diagnose_heads.py answerdotai/ModernBERT-large
"""
import sys

import torch

from decide.data import read
from decide.encoders import TreeMaskedModernBert, tree_mask
from decide.heads import ChoiceHead
from decide.model import collate
from decide.packing import pack
from decide.tokenizer import HFTokenizer

name = sys.argv[1]
torch.manual_seed(0)
tok = HFTokenizer(name)
enc = TreeMaskedModernBert.from_pretrained(name)
enc.resize(len(tok))
enc = enc.cuda().eval()
recs = [r for r in read("data/mix/train.jsonl") if r.task == "ag_news"][:3000]
Q, O, V, Y = [], [], [], []
with torch.no_grad():
    for i in range(0, len(recs), 16):
        chunk = recs[i: i + 16]
        packs = [pack(tok, r.state, r.questions[:1], max_state=512) for r in chunk]
        b = collate(packs, tok.pad_id)
        h = enc(b.ids.cuda(), b.positions.cuda(), tree_mask(b.qgroup, b.ogroup).cuda()).float()
        for j, (p, r) in enumerate(zip(packs, chunk)):
            m = p.opt_markers[0]
            o = torch.zeros(4, h.shape[-1], device="cuda")
            o[: len(m)] = h[j, m]
            Q.append(h[j, p.q_markers[0]])
            O.append(o)
            V.append(torch.tensor([True] * len(m) + [False] * (4 - len(m)), device="cuda"))
            Y.append(r.answers[0])
Q, O, V, Y = torch.stack(Q), torch.stack(O), torch.stack(V), torch.tensor(Y, device="cuda")
tr, te = slice(0, 2500), slice(2500, 3000)
mu, sd = torch.cat([Q[tr], O[tr][V[tr]]]).mean(0), torch.cat([Q[tr], O[tr][V[tr]]]).std(0) + 1e-6
for label, f in (("raw", lambda x: x), ("standardised", lambda x: (x - mu) / sd)):
    torch.manual_seed(0)
    head = ChoiceHead(Q.shape[-1], 16).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for step in range(400):
        idx = torch.randint(0, 2500, (32,), device="cuda")
        logits = head(f(Q[idx]), f(O[idx]) * V[idx][..., None], V[idx])
        loss = torch.nn.functional.cross_entropy(logits, Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                lt = head(f(Q[te]), f(O[te]) * V[te][..., None], V[te])
                acc = (lt.argmax(-1) == Y[te]).float().mean().item()
                print(f"{label:13s} step {step + 1} train loss {loss.item():.3f} held-out slot accuracy {acc:.3f}", flush=True)
