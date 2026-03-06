#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow running as either:
# 1) python -m src.train_gpt_rec
# 2) python src/train_gpt_rec.py
if __package__ in (None, ""):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.data.dataset import (
    TrainChunkDataset,
    build_eval_samples,
    build_train_samples,
    collate_next_item_batch,
)
from src.model.gpt_decoder import GPTRecConfig, GPTRecModel


class _NoOpGradScaler:
    """Fallback scaler for environments without AMP GradScaler support."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def build_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=use_amp)
    return _NoOpGradScaler()


def amp_autocast_context(use_amp: bool):
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
    return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-style next-item recommender on Last.fm sequences.")
    parser.add_argument("--data_dir", type=str, default="data/processed/base")
    parser.add_argument("--output_dir", type=str, default="outputs/runs")
    parser.add_argument("--context_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_k", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def get_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_next_item(
    model: GPTRecModel,
    eval_samples: list[tuple[list[int], int]],
    device: torch.device,
    k: int,
) -> dict:
    model.eval()
    hits = 0.0
    ndcg = 0.0

    for history, target in eval_samples:
        input_ids = torch.tensor(history, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        next_logits = logits[0, -1].clone()
        next_logits[0] = float("-inf")
        topk = torch.topk(next_logits, k=min(k, next_logits.shape[-1])).indices.tolist()

        if target in topk:
            rank = topk.index(target) + 1
            hits += 1.0
            ndcg += 1.0 / math.log2(rank + 1.0)

    n = max(1, len(eval_samples))
    return {f"recall@{k}": hits / n, f"ndcg@{k}": ndcg / n}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = os.path.abspath(args.data_dir)
    train_path = os.path.join(data_dir, "sequences_train.jsonl")
    val_path = os.path.join(data_dir, "sequences_val.jsonl")
    meta = read_meta(data_dir)
    vocab_size = int(meta["num_items_including_pad"])

    train_samples = build_train_samples(
        train_jsonl=train_path,
        context_len=args.context_len,
        stride=args.stride,
    )
    eval_samples = build_eval_samples(
        train_jsonl=train_path,
        target_jsonl=val_path,
        context_len=args.context_len,
    )
    train_ds = TrainChunkDataset(train_samples)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_next_item_batch, pad_id=0),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GPTRecConfig(
        vocab_size=vocab_size,
        max_seq_len=args.context_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = GPTRecModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = device.type == "cuda"
    scaler = build_grad_scaler(use_amp=use_amp)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.abspath(args.output_dir), run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "model": asdict(cfg)}, f, indent=2)

    total_steps = max(1, args.epochs * max(1, len(train_dl)))
    global_step = 0
    best_recall = -1.0

    print(f"[info] device={device} train_samples={len(train_samples):,} eval_users={len(eval_samples):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            global_step += 1
            lr_t = get_lr(global_step, total_steps, args.lr, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_t

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast_context(use_amp=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_dl))
        metrics = evaluate_next_item(model=model, eval_samples=eval_samples, device=device, k=args.eval_k)
        recall_k = metrics[f"recall@{args.eval_k}"]
        ndcg_k = metrics[f"ndcg@{args.eval_k}"]
        print(
            f"[epoch {epoch:02d}] train_loss={avg_loss:.4f} "
            f"recall@{args.eval_k}={recall_k:.4f} ndcg@{args.eval_k}={ndcg_k:.4f}"
        )

        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "args": vars(args),
                "model_cfg": asdict(cfg),
            },
            epoch_ckpt,
        )
        if recall_k > best_recall:
            best_recall = recall_k
            best_ckpt = os.path.join(ckpt_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                    "model_cfg": asdict(cfg),
                },
                best_ckpt,
            )

    print(f"[done] artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
