"""Training and inference of decide-400m on one GPU (``scripts/train.py``, ``scripts/evaluate.py``).

The CPU toy uses ``train.fit``; this module is the full-scale path:

* records are tokenised once (every distinct string encoded in one batched call) and packed into int32 arrays;
* an optimizer step takes ``batch_size`` states; they are grouped by length and split into micro-batches of at most
  ``token_budget`` padded tokens, whose gradients are accumulated, so long and short states cost what they need;
* ModernBERT runs under bf16 autocast with fp32 master weights, layer checkpointing, and an 8-bit AdamW
  (bitsandbytes), which together keep a full fine-tune of the 395M encoder within about 6 GB;
* ``torch.cuda.set_per_process_memory_fraction`` caps the process's allocations (``gpu_memory_gb``), so a run that
  would use more fails instead of crowding other jobs on a shared GPU;
* every ``log_every`` steps a JSON line records the loss of each kind, the learning rate, throughput and peak
  memory; every ``eval_every`` steps, the loss and accuracy on held-back training-task states (``dev``).

Layouts: ``tree`` (decide-400m), ``flat`` (one ordinary pass, options see each other and their order) and
``separate`` (one pass per option). ``none_slot=False`` trains without the *none fits* slot, dropping the choice
questions no option fits, as an ordinary classifier would be trained.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from .data import Record, read
from .model import Batch, DecisionModel, Logits, collate, collate_separate, loss_terms
from .packing import Packed, Separate, pack, pack_separate


@dataclass
class RunConfig:
    name: str = "decide-400m"
    backbone: str = "answerdotai/ModernBERT-large"
    layout: str = "tree"  # tree | flat | separate
    none_slot: bool = True
    max_state: int = 512
    max_question: int = 64
    max_option: int = 32
    epochs: float = 1.0
    batch_size: int = 32  # states per optimizer step
    token_budget: int = 6144  # padded tokens per micro-batch
    lr: float = 3e-5  # encoder
    head_lr: float = 1e-4  # heads, freshly initialised
    weight_decay: float = 0.01
    warmup: float = 0.06
    optimizer: str = "adamw8bit"  # or "adamw"
    checkpointing: bool = True
    bf16: bool = True
    seed: int = 0
    train_limit: int = 0  # 0: every training state; n: the first n of a seeded shuffle (matched-budget ablations)
    gpu_memory_gb: float = 6.2
    max_steps: int = 0  # 0: the whole schedule; n: stop after n steps (smoke tests)
    log_every: int = 50
    eval_every: int = 500
    train: str = "data/mix/train.jsonl"
    dev: str = "data/mix/dev.jsonl"
    alphas: list = field(default_factory=lambda: [0.05, 0.10])
    delta: float = 0.1

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "RunConfig":
        raw = {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**raw)


class CachedTokenizer:
    """An ``HFTokenizer`` whose ``encode`` reads from a table filled by one batched call over every distinct string."""

    def __init__(self, tok, texts: Sequence[str] = (), max_len: int = 512):
        self.tok = tok
        self.cls_id, self.pad_id, self.q_id, self.opt_id = tok.cls_id, tok.pad_id, tok.q_id, tok.opt_id
        self.kind_ids = tok.kind_ids
        self.table: dict[str, list[int]] = {}
        self.add(texts, max_len)

    def add(self, texts: Sequence[str], max_len: int = 512) -> None:
        new = sorted({t for t in texts if t not in self.table})
        for s in range(0, len(new), 4096):
            chunk = new[s: s + 4096]
            enc = self.tok.tok(chunk, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
            self.table.update(zip(chunk, enc))

    def encode(self, text: str) -> list[int]:
        got = self.table.get(text)
        return got if got is not None else self.tok.encode(text)

    def __len__(self) -> int:
        return len(self.tok)


def texts_of(records: Sequence[Record]) -> list[str]:
    out = []
    for r in records:
        out.append(r.state)
        for q in r.questions:
            out.append(q.text)
            out.extend(q.options)
    return out


def _compact(p: Packed) -> Packed:
    for name in ("ids", "positions", "qgroup", "ogroup"):
        setattr(p, name, np.asarray(getattr(p, name), dtype=np.int32))
    return p


def pack_record(tok, r: Record, cfg: RunConfig):
    kw = dict(max_state=cfg.max_state, max_question=cfg.max_question, max_option=cfg.max_option)
    if cfg.layout == "separate":
        s = pack_separate(tok, r.state, r.questions, **kw)
        s.sequences = [_compact(p) for p in s.sequences]
        return s
    return _compact(pack(tok, r.state, r.questions, layout=cfg.layout, **kw))


def cost(item) -> tuple[int, int]:
    """(rows, longest row) of a packed item."""
    if isinstance(item, Separate):
        return len(item.sequences), max(len(p) for p in item.sequences)
    return 1, len(item)


def drop_none_fits(records: list[Record]) -> list[Record]:
    """The records without their choice questions no option fits (training without the none slot)."""
    out = []
    for r in records:
        keep = [i for i, (q, a) in enumerate(zip(r.questions, r.answers)) if not (q.kind == "choice" and a == 0)]
        if keep:
            out.append(Record(r.state, [r.questions[i] for i in keep], [r.answers[i] for i in keep], r.task,
                              [r.tags[i] for i in keep] if r.tags else []))
    return out


def micro_batches(items: Sequence, order: Sequence[int], budget: int) -> list[list[int]]:
    """Splits ``order`` (sorted by length) into groups whose padded size, rows x longest row, fits ``budget``."""
    groups, cur, rows, longest = [], [], 0, 0
    for i in order:
        r, n = cost(items[i])
        if cur and (rows + r) * max(longest, n) > budget:
            groups.append(cur)
            cur, rows, longest = [], 0, 0
        cur.append(i)
        rows += r
        longest = max(longest, n)
    if cur:
        groups.append(cur)
    return groups


def steps_of(items: Sequence, cfg: RunConfig, rng: random.Random) -> list[list[list[int]]]:
    """One epoch: optimizer steps of ``batch_size`` states, each as micro-batches. States are grouped by length
    inside windows of 64 steps, and the steps are then shuffled."""
    idx = list(range(len(items)))
    rng.shuffle(idx)
    window = cfg.batch_size * 64
    steps = []
    for s in range(0, len(idx), window):
        chunk = sorted(idx[s: s + window], key=lambda i: cost(items[i])[0] * cost(items[i])[1])
        for b in range(0, len(chunk), cfg.batch_size):
            batch = chunk[b: b + cfg.batch_size]
            steps.append(micro_batches(items, batch, cfg.token_budget))
    rng.shuffle(steps)
    return steps


def make_batch(items: Sequence, group: Sequence[int], pad_id: int) -> Batch:
    chosen = [items[i] for i in group]
    if isinstance(chosen[0], Separate):
        return collate_separate(chosen, pad_id)
    return collate(chosen, pad_id)


def targets_for(records: Sequence[Record], group: Sequence[int], where: dict, device) -> dict[str, torch.Tensor]:
    return {k: torch.tensor([records[group[b]].answers[i] for b, i in rows], dtype=torch.long, device=device)
            for k, rows in where.items()}


def build_model(cfg: RunConfig, vocab: int, device: str) -> DecisionModel:
    from .encoders import TreeMaskedModernBert

    encoder = TreeMaskedModernBert.from_pretrained(cfg.backbone, checkpointing=cfg.checkpointing)
    encoder.resize(vocab)
    return DecisionModel(encoder, heads=16, none_slot=cfg.none_slot).to(device)


def optimizer_for(model: DecisionModel, cfg: RunConfig):
    enc_decay, enc_plain, heads = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            (enc_decay if p.ndim >= 2 and "embeddings" not in name else enc_plain).append(p)
        else:
            heads.append(p)
    groups = [{"params": enc_decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
              {"params": enc_plain, "lr": cfg.lr, "weight_decay": 0.0},
              {"params": heads, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay}]
    if cfg.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999), eps=1e-8)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def autocast(cfg: RunConfig, device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and device == "cuda")


@torch.no_grad()
def predict(model: DecisionModel, items: Sequence, records: Sequence[Record], cfg: RunConfig, device: str,
            pad_id: int, budget: int | None = None) -> list[dict]:
    """Raw outputs of every question, in record order: per question a dict with the record index, question index,
    kind, target and logits (a choice's 1 + K slot logits, a yes/no's logit, or a score's eta and cutpoints)."""
    model.eval()
    order = sorted(range(len(items)), key=lambda i: cost(items[i])[0] * cost(items[i])[1])
    out: list[dict] = []
    for group in micro_batches(items, order, budget or cfg.token_budget * 4):
        batch = make_batch(items, group, pad_id).to(device)
        with autocast(cfg, device):
            logits = model(batch)
        out += rows_of(logits, records, group)
    out.sort(key=lambda r: (r["record"], r["question"]))
    return out


def rows_of(logits: Logits, records: Sequence[Record], group: Sequence[int]) -> list[dict]:
    rows = []
    ch, yn, eta, cuts = (logits.choice.float().cpu(), logits.yesno.float().cpu(), logits.eta.float().cpu(),
                         logits.cuts.float().cpu())
    for kind, where in logits.where.items():
        for r, (b, i) in enumerate(where):
            rec = records[group[b]]
            q = rec.questions[i]
            row = {"record": group[b], "question": i, "kind": kind, "target": rec.answers[i]}
            if kind == "choice":
                row["logits"] = ch[r, : 1 + len(q.options)].tolist()
            elif kind == "yesno":
                row["logits"] = [yn[r].item()]
            else:
                row["logits"] = [eta[r].item()] + cuts[r, : len(q.options) - 1].tolist()
            rows.append(row)
    return rows


def dev_metrics(rows: list[dict]) -> dict:
    """Loss and accuracy of each kind from ``predict`` rows (temperature 1, the most probable slot or level)."""
    from .heads import level_probs

    res = {}
    for kind in ("choice", "yesno", "score"):
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        nll, right = 0.0, 0
        for r in sel:
            z = torch.tensor(r["logits"])
            if kind == "choice":
                lp = torch.log_softmax(z, -1)
                nll -= lp[r["target"]].item()
                right += int(lp.argmax().item() == r["target"])
            elif kind == "yesno":
                p = torch.sigmoid(z[0]).item()
                nll -= math.log(max(p if r["target"] else 1 - p, 1e-9))
                right += int((p >= 0.5) == bool(r["target"]))
            else:
                p = level_probs(z[:1], z[1:][None])[0]
                nll -= math.log(max(p[r["target"]].item(), 1e-9))
                right += int(p.argmax().item() == r["target"])
        res[kind] = {"loss": nll / len(sel), "accuracy": right / len(sel), "n": len(sel)}
    return res


def machine() -> dict:
    import platform

    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    info = {"cpu": cpu, "torch": torch.__version__, "python": platform.python_version()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import bitsandbytes

        info["bitsandbytes"] = bitsandbytes.__version__
    except ImportError:
        pass
    return info


def train(cfg: RunConfig, out: Path, log: Callable[[str], None] = print) -> dict:
    from .tokenizer import HFTokenizer

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1) + "\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and cfg.gpu_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg.gpu_memory_gb * 2**30 / total))
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    t0 = time.time()
    records = list(read(cfg.train))
    if cfg.train_limit:
        order = list(range(len(records)))
        random.Random(1234).shuffle(order)  # the same subset for every seed and variant
        records = [records[i] for i in sorted(order[: cfg.train_limit])]
    if not cfg.none_slot:
        records = drop_none_fits(records)
    dev = list(read(cfg.dev))
    tok = CachedTokenizer(HFTokenizer(cfg.backbone), texts_of(records) + texts_of(dev), cfg.max_state)
    items = [pack_record(tok, r, cfg) for r in records]
    dev_items = [pack_record(tok, r, cfg) for r in dev]
    log(f"{len(records)} training states, {sum(len(r.questions) for r in records)} questions; packed in "
        f"{time.time() - t0:.0f} s")
    model = build_model(cfg, len(tok), device)
    params = sum(p.numel() for p in model.parameters())
    opt = optimizer_for(model, cfg)
    epochs = max(1, math.ceil(cfg.epochs))
    plan = [steps_of(items, cfg, rng) for _ in range(epochs)]
    total = int(round(len(plan[0]) * cfg.epochs))
    plan = [s for epoch in plan for s in epoch][:total]
    if cfg.max_steps:
        plan = plan[: cfg.max_steps]
    warm = max(1, int(cfg.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))
    logf = open(out / "log.jsonl", "w")

    def record(row: dict) -> None:
        logf.write(json.dumps(row) + "\n")
        logf.flush()

    record({"event": "start", "machine": machine(), "parameters": params, "steps": total, "states": len(records),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    run = {"loss": {}, "n": {}, "tokens": 0, "padded": 0, "states": 0}
    tick, train_start, tokens_total = time.time(), time.time(), 0
    last = len(plan)
    for step, micro in enumerate(plan, 1):
        model.train()
        nq = sum(len(records[i].questions) for g in micro for i in g)
        for group in micro:
            batch = make_batch(items, group, tok.pad_id).to(device)
            with autocast(cfg, device):
                logits = model(batch)
            terms = loss_terms(logits, targets_for(records, group, logits.where, device))
            value = torch.stack([v for v, _ in terms.values()]).sum() / nq
            value.backward()
            for k, (v, n) in terms.items():
                run["loss"][k] = run["loss"].get(k, 0.0) + v.item()
                run["n"][k] = run["n"].get(k, 0) + n
            run["tokens"] += int((batch.qgroup != -2).sum())
            run["padded"] += batch.ids.numel()
            run["states"] += len(group)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % cfg.log_every == 0 or step == last:
            dt = time.time() - tick
            tokens_total += run["tokens"]
            row = {"event": "train", "step": step, "of": total, "lr": sched.get_last_lr()[0],
                   "loss": {k: run["loss"][k] / run["n"][k] for k in run["loss"]},
                   "tokens_per_s": run["tokens"] / dt, "states_per_s": run["states"] / dt,
                   "padding": 1 - run["tokens"] / max(run["padded"], 1), "elapsed_s": time.time() - train_start,
                   "peak_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0}
            record(row)
            log(f"step {step}/{total} " + " ".join(f"{k} {v:.3f}" for k, v in row["loss"].items())
                + f" | {row['tokens_per_s']:.0f} tok/s, peak {row['peak_gb']:.2f} GB, {row['elapsed_s'] / 60:.1f} min")
            run = {"loss": {}, "n": {}, "tokens": 0, "padded": 0, "states": 0}
            tick = time.time()
        if step % cfg.eval_every == 0 or step == last:
            metrics = dev_metrics(predict(model, dev_items, dev, cfg, device, tok.pad_id))
            record({"event": "dev", "step": step, **metrics})
            log(f"dev @ {step}: " + " ".join(f"{k} loss {m['loss']:.3f} acc {m['accuracy']:.3f}" for k, m in metrics.items()))
    train_s = time.time() - train_start
    torch.save(model.state_dict(), out / "model.pt")
    summary = {"name": cfg.name, "seed": cfg.seed, "steps": last, "states": len(records), "train_seconds": train_s,
               "tokens": tokens_total, "parameters": params, "machine": machine(),
               "peak_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0,
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    record({"event": "end", **summary})
    logf.close()
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"trained in {train_s / 60:.1f} min; wrote {out / 'model.pt'}")
    return summary
