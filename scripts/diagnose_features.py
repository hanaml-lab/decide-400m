"""Diagnostic: are ModernBERT's outputs at the marker tokens informative before any training? Compares the tree
wrapper with the library's forward on an ordinary sequence, measures how alike the marker vectors of different
states are, and fits a linear probe (logistic regression) for AG News topics on each kind of vector.

    python scripts/diagnose_features.py answerdotai/ModernBERT-large
"""
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from decide.data import read
from decide.encoders import TreeMaskedModernBert, tree_mask
from decide.model import collate
from decide.packing import pack
from decide.tasks import AG_LABELS
from decide.tokenizer import HFTokenizer

name = sys.argv[1]
tok = HFTokenizer(name)
enc = TreeMaskedModernBert.from_pretrained(name)
enc.resize(len(tok))
enc = enc.cuda().eval()
recs = [r for r in read("data/mix/train.jsonl") if r.task == "ag_news"][:1200]
ids = torch.tensor([tok.tok("A plain sentence about the markets and the weather today.")["input_ids"]]).cuda()
with torch.no_grad():
    ref = enc.model(input_ids=ids).last_hidden_state
    full = torch.ones(1, ids.shape[1], ids.shape[1], dtype=torch.bool, device="cuda")
    ours = enc(ids, torch.arange(ids.shape[1], device="cuda")[None], full)
print("wrapper vs library, max abs difference:", (ours - ref).abs().max().item(), "| output scale:", ref.abs().mean().item())
feats = {"Q marker": [], "first OPT marker": [], "mean of state tokens": [], "CLS": []}
with torch.no_grad():
    for i in range(0, len(recs), 16):
        chunk = recs[i: i + 16]
        packs = [pack(tok, r.state, r.questions[:1], max_state=512) for r in chunk]
        b = collate(packs, tok.pad_id)
        h = enc(b.ids.cuda(), b.positions.cuda(), tree_mask(b.qgroup, b.ogroup).cuda()).float().cpu()
        for j, (p, r) in enumerate(zip(packs, chunk)):
            feats["Q marker"].append(h[j, p.q_markers[0]].numpy())
            feats["first OPT marker"].append(h[j, p.opt_markers[0][0]].numpy())
            n_state = sum(1 for g in p.qgroup if g == -1)
            feats["mean of state tokens"].append(h[j, 1:n_state].mean(0).numpy())
            feats["CLS"].append(h[j, 0].numpy())
y = []  # the AG News topic of each state, from its choice question's answer (-1 when no option fits)
for r in recs:
    q, a = r.questions[0], r.answers[0]
    y.append(AG_LABELS.index(q.options[a - 1]) if a else -1)
y = np.array(y)
keep = y >= 0
for k, v in feats.items():
    X = np.array(v)[keep]
    yy = y[keep]
    sims = X @ X.T / (np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(X, axis=1)[None])
    off = sims[~np.eye(len(X), dtype=bool)].mean()
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xs[:700], yy[:700])
    print(f"{k:22s} mean cosine between states {off:.3f} | probe accuracy on held-out states {clf.score(Xs[700:], yy[700:]):.3f}")
