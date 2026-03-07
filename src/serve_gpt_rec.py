#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import asdict
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Allow running as either:
# 1) python -m src.serve_gpt_rec
# 2) python src/serve_gpt_rec.py
if __package__ in (None, ""):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.model.gpt_decoder import GPTRecConfig, GPTRecModel


class PredictRequest(BaseModel):
    history_artist_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)


class ModelServer:
    def __init__(self, checkpoint_path: str, data_dir: str, device: str) -> None:
        self.data_dir = os.path.abspath(data_dir)
        self.device = torch.device(device)
        self.vocab, self.token_to_artist = self._load_vocab(self.data_dir)
        self.artist_ids = sorted(self.vocab.keys())
        self.model, self.model_cfg = self._load_model(checkpoint_path, self.device)

    def _load_vocab(self, data_dir: str) -> tuple[dict[str, int], dict[int, str]]:
        vocab_path = os.path.join(data_dir, "artist_vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"artist_vocab.json not found at: {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
        vocab = {str(k): int(v) for k, v in vocab_raw.items()}
        token_to_artist = {token: artist for artist, token in vocab.items()}
        return vocab, token_to_artist

    def _load_model(self, checkpoint_path: str, device: torch.device) -> tuple[GPTRecModel, GPTRecConfig]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg_dict = checkpoint.get("model_cfg")
        if cfg_dict is None:
            raise ValueError("Checkpoint is missing 'model_cfg'.")
        cfg = GPTRecConfig(**cfg_dict)
        model = GPTRecModel(cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, cfg

    @torch.no_grad()
    def predict(self, history_artist_ids: list[str], top_k: int) -> dict[str, Any]:
        known_tokens = [self.vocab[a] for a in history_artist_ids if a in self.vocab]
        unknown_artist_ids = [a for a in history_artist_ids if a not in self.vocab]
        if not known_tokens:
            raise HTTPException(
                status_code=400,
                detail="No known artists found in history_artist_ids. Use /artists to search valid ids.",
            )

        known_tokens = known_tokens[-self.model_cfg.max_seq_len :]
        input_ids = torch.tensor(known_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        next_logits = logits[0, -1].clone()
        next_logits[0] = float("-inf")  # never recommend PAD
        k = min(top_k, next_logits.shape[-1])
        top_scores, top_tokens = torch.topk(next_logits, k=k)

        probs = torch.softmax(next_logits, dim=-1)
        predictions: list[dict[str, Any]] = []
        for token, score in zip(top_tokens.tolist(), top_scores.tolist(), strict=False):
            artist_id = self.token_to_artist.get(int(token), f"<unk-token-{token}>")
            predictions.append(
                {
                    "artist_id": artist_id,
                    "token_id": int(token),
                    "logit": float(score),
                    "prob": float(probs[token].item()),
                }
            )

        used_artist_ids = [self.token_to_artist[t] for t in known_tokens if t in self.token_to_artist]
        return {
            "history_used_artist_ids": used_artist_ids,
            "unknown_artist_ids": unknown_artist_ids,
            "top_k": k,
            "predictions": predictions,
        }

    def search_artists(self, query: str, limit: int) -> list[dict[str, Any]]:
        q = query.strip().lower()
        if not q:
            candidates = self.artist_ids[:limit]
        else:
            starts = [a for a in self.artist_ids if a.lower().startswith(q)]
            contains = [a for a in self.artist_ids if q in a.lower() and not a.lower().startswith(q)]
            candidates = (starts + contains)[:limit]

        return [{"artist_id": a, "token_id": int(self.vocab[a])} for a in candidates]


def find_latest_checkpoint(outputs_dir: str) -> str | None:
    pattern = os.path.join(os.path.abspath(outputs_dir), "*", "checkpoints", "best.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def build_app(server: ModelServer, web_dir: str) -> FastAPI:
    app = FastAPI(title="Last.fm GPT Recommender", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/meta")
    def meta() -> dict[str, Any]:
        return {
            "device": str(server.device),
            "model_cfg": asdict(server.model_cfg),
            "num_artists": len(server.artist_ids),
        }

    @app.get("/artists")
    def artists(
        q: str = Query(default="", description="Search query"),
        limit: int = Query(default=20, ge=1, le=200),
    ) -> dict[str, Any]:
        return {"results": server.search_artists(query=q, limit=limit)}

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        return server.predict(history_artist_ids=req.history_artist_ids, top_k=req.top_k)

    if os.path.isdir(web_dir):
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GPT recommender with a simple interactive web UI.")
    parser.add_argument("--data_dir", type=str, default="data/processed/base")
    parser.add_argument("--outputs_dir", type=str, default="outputs/runs")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def pick_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.outputs_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Train first or pass --checkpoint /path/to/best.pt."
        )
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = pick_device(args.device)
    server = ModelServer(
        checkpoint_path=checkpoint_path,
        data_dir=args.data_dir,
        device=device,
    )
    web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
    app = build_app(server=server, web_dir=web_dir)

    print(f"[info] serving model on http://{args.host}:{args.port}")
    print(f"[info] checkpoint: {checkpoint_path}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
