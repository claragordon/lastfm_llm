#!/usr/bin/env python3
"""
Last.fm 1K preprocessing (artist-level sequential rec)

Input:
  userid-timestamp-artid-artname-traid-traname.tsv

Outputs (in --out_dir):
  - meta.json
  - artist_vocab.json              (original_artist_id -> token_id)
  - sequences_train.jsonl          (one user per line)
  - sequences_val.jsonl
  - sequences_test.jsonl

Each JSONL line:
  {"user": "<user_id>", "seq": [int, int, ...]}

Token IDs:
  0 is reserved for PAD (not used inside seqs here, but convenient later).
  Artists start from 1.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Config:
    input_tsv: str
    out_dir: str
    min_artist_freq: int
    min_user_events: int
    max_events_per_user: int | None
    collapse_consecutive_repeats: bool
    keep_most_recent_if_tie: bool
    chunksize: int


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--input_tsv", required=True, help="Path to userid-timestamp-artid-...tsv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_artist_freq", type=int, default=20)
    p.add_argument("--min_user_events", type=int, default=20)
    p.add_argument("--max_events_per_user", type=int, default=0,
                   help="If >0, keep only the most recent N events per user (after sorting).")
    p.add_argument("--collapse_consecutive_repeats", action="store_true",
                   help="Collapse consecutive identical artists within each user sequence.")
    p.add_argument("--keep_most_recent_if_tie", action="store_true",
                   help="If multiple events share same timestamp, keep last seen ordering (stable sort).")
    p.add_argument("--chunksize", type=int, default=1_000_000)
    a = p.parse_args()

    return Config(
        input_tsv=a.input_tsv,
        out_dir=a.out_dir,
        min_artist_freq=a.min_artist_freq,
        min_user_events=a.min_user_events,
        max_events_per_user=(a.max_events_per_user if a.max_events_per_user > 0 else None),
        collapse_consecutive_repeats=bool(a.collapse_consecutive_repeats),
        keep_most_recent_if_tie=bool(a.keep_most_recent_if_tie),
        chunksize=a.chunksize,
    )


def collapse_repeats(seq: List[int]) -> List[int]:
    if not seq:
        return seq
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def iter_chunks(cfg: Config) -> Iterable[pd.DataFrame]:
    # Original TSV columns:
    # 0 user, 1 timestamp, 2 artist_id, 3 artist_name, 4 track_id, 5 track_name
    cols = ["user", "ts", "artist"]
    # ts is ISO 8601 like 2009-04-08T01:57:47Z
    return pd.read_csv(
        cfg.input_tsv,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=cols,
        dtype={"user": "string", "ts": "string", "artist": "string"},
        chunksize=cfg.chunksize,
        encoding_errors="ignore",
        on_bad_lines="skip",
    )


def first_pass_counts(cfg: Config) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Count artist freq and user event counts without loading full dataset."""
    artist_counts: defaultdict[str, int] = defaultdict(int)
    user_counts: defaultdict[str, int] = defaultdict(int)

    for i, chunk in enumerate(iter_chunks(cfg), start=1):
        # drop obvious nulls
        chunk = chunk.dropna(subset=["user", "ts", "artist"])
        # update counts
        for a, c in chunk["artist"].value_counts().items():
            artist_counts[str(a)] += int(c)
        for u, c in chunk["user"].value_counts().items():
            user_counts[str(u)] += int(c)

        if i % 10 == 0:
            print(f"[pass1] processed ~{i*cfg.chunksize:,} rows")

    return artist_counts, user_counts


def second_pass_collect(cfg: Config, valid_artists: set[str], valid_users: set[str]) -> pd.DataFrame:
    """Collect filtered data into a DataFrame (still can be large, but manageable after filtering)."""
    parts: List[pd.DataFrame] = []
    for i, chunk in enumerate(iter_chunks(cfg), start=1):
        chunk = chunk.dropna(subset=["user", "ts", "artist"])
        chunk = chunk[chunk["artist"].isin(valid_artists) & chunk["user"].isin(valid_users)]
        if len(chunk) == 0:
            continue
        parts.append(chunk)

        if i % 10 == 0:
            print(f"[pass2] processed ~{i*cfg.chunksize:,} rows, kept so far ~{sum(len(p) for p in parts):,}")

    if not parts:
        raise RuntimeError("No rows left after filtering. Lower thresholds or verify input format.")
    df = pd.concat(parts, ignore_index=True)
    return df


def build_sequences(cfg: Config, df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return user -> list of artist IDs (strings), sorted by timestamp."""
    # Parse timestamps. Coerce errors to NaT then drop.
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])

    # Stable sort so ties keep file/chunk order if desired.
    kind = "mergesort" if cfg.keep_most_recent_if_tie else "quicksort"
    df = df.sort_values(["user", "ts"], kind=kind)

    # Group to sequences
    user_to_artists: Dict[str, List[str]] = {}
    for user, g in df.groupby("user", sort=False):
        seq = [str(x) for x in g["artist"].tolist()]
        if cfg.max_events_per_user is not None and len(seq) > cfg.max_events_per_user:
            seq = seq[-cfg.max_events_per_user :]  # keep most recent
        user_to_artists[str(user)] = seq

    return user_to_artists


def filter_and_postprocess_sequences(
    cfg: Config,
    user_to_artists: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Apply collapse repeats (optional) and min_user_events after that transformation."""
    out: Dict[str, List[str]] = {}
    for u, seq in user_to_artists.items():
        if cfg.collapse_consecutive_repeats:
            # collapse is defined for ints; do for strings here
            collapsed: List[str] = []
            prev = None
            for a in seq:
                if a != prev:
                    collapsed.append(a)
                prev = a
            seq = collapsed
        if len(seq) >= cfg.min_user_events:
            out[u] = seq
    return out


def make_vocab(user_to_artists: Dict[str, List[str]]) -> Dict[str, int]:
    """Map original artist_id (string) -> token_id (int). Reserve 0 for PAD."""
    # Deterministic ordering: sort by frequency then by id to stabilize across runs.
    freq: Dict[str, int] = {}
    for seq in user_to_artists.values():
        for a in seq:
            freq[a] = freq.get(a, 0) + 1

    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab: Dict[str, int] = {}
    next_id = 1
    for a, _ in items:
        vocab[a] = next_id
        next_id += 1
    return vocab


def apply_vocab(user_to_artists: Dict[str, List[str]], vocab: Dict[str, int]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for u, seq in user_to_artists.items():
        out[u] = [vocab[a] for a in seq if a in vocab]
    return out


def split_leave_last_two(user_to_seq: Dict[str, List[int]]) -> Tuple[Dict[str, List[int]], Dict[str, int], Dict[str, int]]:
    train: Dict[str, List[int]] = {}
    val: Dict[str, int] = {}
    test: Dict[str, int] = {}
    for u, seq in user_to_seq.items():
        if len(seq) < 3:
            continue
        train[u] = seq[:-2]
        val[u] = seq[-2]
        test[u] = seq[-1]
    return train, val, test


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("[info] pass 1: counting artists/users")
    artist_counts, user_counts = first_pass_counts(cfg)

    valid_artists = {a for a, c in artist_counts.items() if c >= cfg.min_artist_freq}
    valid_users = {u for u, c in user_counts.items() if c >= cfg.min_user_events}

    print(f"[info] artists total={len(artist_counts):,}, valid={len(valid_artists):,} (min_freq={cfg.min_artist_freq})")
    print(f"[info] users total={len(user_counts):,}, valid={len(valid_users):,} (min_events={cfg.min_user_events})")

    print("[info] pass 2: collecting filtered rows")
    df = second_pass_collect(cfg, valid_artists, valid_users)
    print(f"[info] kept rows after initial filtering: {len(df):,}")

    print("[info] building per-user sequences (sorted by timestamp)")
    user_to_artists = build_sequences(cfg, df)

    print("[info] optional postprocessing (collapse repeats, min_user_events again)")
    user_to_artists = filter_and_postprocess_sequences(cfg, user_to_artists)
    print(f"[info] users after postprocessing: {len(user_to_artists):,}")

    print("[info] building vocab")
    vocab = make_vocab(user_to_artists)
    user_to_seq = apply_vocab(user_to_artists, vocab)

    # Final stats
    num_users = len(user_to_seq)
    num_items = len(vocab) + 1  # + PAD
    total_events = sum(len(s) for s in user_to_seq.values())
    avg_len = total_events / max(1, num_users)

    print(f"[stats] users={num_users:,} items(vocab+pad)={num_items:,} events={total_events:,} avg_len={avg_len:.1f}")

    print("[info] splitting train/val/test (leave-last-two)")
    train, val, test = split_leave_last_two(user_to_seq)

    # Persist
    meta = {
        "input_tsv": os.path.abspath(cfg.input_tsv),
        "min_artist_freq": cfg.min_artist_freq,
        "min_user_events": cfg.min_user_events,
        "max_events_per_user": cfg.max_events_per_user,
        "collapse_consecutive_repeats": cfg.collapse_consecutive_repeats,
        "keep_most_recent_if_tie": cfg.keep_most_recent_if_tie,
        "chunksize": cfg.chunksize,
        "num_users": num_users,
        "num_items_including_pad": num_items,
        "total_events": total_events,
        "avg_seq_len": avg_len,
    }
    with open(os.path.join(cfg.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(cfg.out_dir, "artist_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    write_jsonl(
        os.path.join(cfg.out_dir, "sequences_train.jsonl"),
        ({"user": u, "seq": seq} for u, seq in train.items()),
    )
    write_jsonl(
        os.path.join(cfg.out_dir, "sequences_val.jsonl"),
        ({"user": u, "seq": [val[u]]} for u in val.keys()),
    )
    write_jsonl(
        os.path.join(cfg.out_dir, "sequences_test.jsonl"),
        ({"user": u, "seq": [test[u]]} for u in test.keys()),
    )

    print(f"[done] wrote outputs to: {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()