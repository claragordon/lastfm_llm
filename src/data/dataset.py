from __future__ import annotations

import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def load_jsonl_sequences(path: str) -> Dict[str, List[int]]:
    sequences: Dict[str, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user = str(row["user"])
            seq = [int(x) for x in row["seq"]]
            sequences[user] = seq
    return sequences


def build_train_samples(train_jsonl: str, context_len: int, stride: int) -> List[List[int]]:
    user_to_seq = load_jsonl_sequences(train_jsonl)
    samples: List[List[int]] = []

    for seq in user_to_seq.values():
        if len(seq) < 2:
            continue

        max_window = context_len + 1
        if len(seq) <= max_window:
            samples.append(seq)
            continue

        for start in range(0, len(seq) - 1, stride):
            window = seq[start : start + max_window]
            if len(window) < 2:
                continue
            samples.append(window)

    return samples


def build_eval_samples(train_jsonl: str, target_jsonl: str, context_len: int) -> List[Tuple[List[int], int]]:
    train_seqs = load_jsonl_sequences(train_jsonl)
    target_seqs = load_jsonl_sequences(target_jsonl)
    samples: List[Tuple[List[int], int]] = []

    for user, target in target_seqs.items():
        history = train_seqs.get(user, [])
        if len(history) == 0 or len(target) == 0:
            continue
        samples.append((history[-context_len:], int(target[0])))

    return samples


class TrainChunkDataset(Dataset[List[int]]):
    def __init__(self, samples: List[List[int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List[int]:
        return self.samples[idx]


def collate_next_item_batch(batch: List[List[int]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(len(seq) for seq in batch)
    input_ids: List[List[int]] = []
    labels: List[List[int]] = []
    attention_masks: List[List[int]] = []

    for seq in batch:
        x = seq[:-1]
        y = seq[1:]
        pad_len = (max_len - 1) - len(x)
        input_ids.append(x + [pad_id] * pad_len)
        labels.append(y + [-100] * pad_len)
        attention_masks.append([1] * len(x) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
    }
