#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import certifi
except ModuleNotFoundError:
    certifi = None

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
    include_attention: bool = False
    attention_top_n: int = Field(default=5, ge=1, le=20)


class ModelServer:
    def __init__(self, checkpoint_path: str, data_dir: str, device: str) -> None:
        self.data_dir = os.path.abspath(data_dir)
        self.device = torch.device(device)
        self.vocab, self.token_to_artist = self._load_vocab(self.data_dir)
        self.artist_names = self._load_artist_names(self.data_dir)
        self.artist_images = self._load_artist_images(self.data_dir)
        self._musicbrainz_last_request_ts = 0.0
        self._image_retry_seconds = 24 * 60 * 60
        self._image_retry_seconds_transient = 5 * 60
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

    def _load_artist_names(self, data_dir: str) -> dict[str, str]:
        path = os.path.join(data_dir, "artist_names.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(artist_id): str(name) for artist_id, name in raw.items()}

    def _artist_name(self, artist_id: str) -> str:
        return self.artist_names.get(artist_id, artist_id)

    def _load_artist_images(self, data_dir: str) -> dict[str, dict[str, Any]]:
        path = os.path.join(data_dir, "artist_images.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        parsed: dict[str, dict[str, Any]] = {}
        for artist_id, value in raw.items():
            artist_id = str(artist_id)
            if isinstance(value, dict):
                parsed[artist_id] = {
                    "url": str(value.get("url")) if value.get("url") else None,
                    "last_checked": float(value.get("last_checked", 0.0)),
                    "last_error": str(value.get("last_error")) if value.get("last_error") else None,
                }
                continue
            # Backward compatibility for old flat format: {"artist_id": "url"|null}
            parsed[artist_id] = {
                "url": str(value) if value else None,
                "last_checked": 0.0,
                "last_error": None,
            }
        return parsed

    def _save_artist_images(self) -> None:
        path = os.path.join(self.data_dir, "artist_images.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.artist_images, f)

    def _fetch_json(self, url: str) -> dict[str, Any]:
        now = time.time()
        delta = now - self._musicbrainz_last_request_ts
        if delta < 1.0:
            time.sleep(1.0 - delta)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "lastfm-llm/0.1 (local demo app)",
                "Accept": "application/json",
            },
        )
        ssl_context = None
        if certifi is not None:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=8, context=ssl_context) as resp:
            self._musicbrainz_last_request_ts = time.time()
            payload = resp.read().decode("utf-8")
        return json.loads(payload)

    def _extract_image_url_from_musicbrainz_artist(self, artist_payload: dict[str, Any]) -> str | None:
        relations = artist_payload.get("relations", [])
        for rel in relations:
            rel_type = str(rel.get("type", "")).lower()
            url_info = rel.get("url", {})
            resource = url_info.get("resource")
            if rel_type == "image" and isinstance(resource, str):
                # Some relations point to Wikimedia file pages; convert to a direct file path URL.
                if "commons.wikimedia.org/wiki/File:" in resource:
                    filename = resource.split("/wiki/File:", 1)[-1]
                    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(filename)}"
                return resource

        wikidata_url = None
        for rel in relations:
            rel_type = str(rel.get("type", "")).lower()
            url_info = rel.get("url", {})
            resource = url_info.get("resource")
            if rel_type == "wikidata" and isinstance(resource, str):
                wikidata_url = resource
                break
        if wikidata_url is None:
            return None

        qid = wikidata_url.rstrip("/").split("/")[-1]
        if not qid:
            return None
        wikidata_api = (
            "https://www.wikidata.org/w/api.php?action=wbgetentities&ids="
            f"{urllib.parse.quote(qid)}&props=claims&format=json"
        )
        wikidata_payload = self._fetch_json(wikidata_api)
        entities = wikidata_payload.get("entities", {})
        entity = entities.get(qid, {})
        claims = entity.get("claims", {})
        image_claims = claims.get("P18", [])
        if not image_claims:
            return None
        filename = (
            image_claims[0]
            .get("mainsnak", {})
            .get("datavalue", {})
            .get("value")
        )
        if not isinstance(filename, str) or not filename:
            return None
        return f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(filename)}"

    def _fetch_artist_image_from_musicbrainz(self, artist_name: str) -> str | None:
        artist_name = artist_name.strip()
        if not artist_name:
            return None
        search_url = (
            "https://musicbrainz.org/ws/2/artist/?query=artist:"
            f"{urllib.parse.quote(artist_name)}&fmt=json&limit=1"
        )
        search_payload = self._fetch_json(search_url)
        artists = search_payload.get("artists", [])
        if not artists:
            return None
        mbid = artists[0].get("id")
        if not isinstance(mbid, str) or not mbid:
            return None
        detail_url = f"https://musicbrainz.org/ws/2/artist/{urllib.parse.quote(mbid)}?inc=url-rels&fmt=json"
        detail_payload = self._fetch_json(detail_url)
        return self._extract_image_url_from_musicbrainz_artist(detail_payload)

    def _artist_image_url(self, artist_id: str, lazy: bool = True) -> str | None:
        entry = self.artist_images.get(artist_id)
        now = time.time()
        if entry is not None:
            if entry.get("url"):
                return str(entry["url"])
            # Retry misses periodically rather than caching null forever.
            last_checked = float(entry.get("last_checked", 0.0))
            last_error = str(entry.get("last_error") or "")
            retry_seconds = self._image_retry_seconds
            # Old TLS failures are stale after certifi fix; retry immediately.
            if "CERTIFICATE_VERIFY_FAILED" in last_error:
                retry_seconds = 0
            # Treat 5xx upstream issues as transient.
            if "HTTP Error 503" in last_error or "HTTP Error 429" in last_error:
                retry_seconds = self._image_retry_seconds_transient
            if not lazy or (now - last_checked) < retry_seconds:
                return None
        if not lazy:
            return None
        artist_name = self._artist_name(artist_id)
        try:
            image_url = self._fetch_artist_image_from_musicbrainz(artist_name)
            last_error = None
        except Exception as exc:
            print(f"[warn] artist image fetch failed for '{artist_name}' ({artist_id}): {exc}")
            image_url = None
            last_error = str(exc)
        self.artist_images[artist_id] = {
            "url": image_url,
            "last_checked": now,
            "last_error": last_error,
        }
        self._save_artist_images()
        return image_url

    @torch.no_grad()
    def predict(
        self,
        history_artist_ids: list[str],
        top_k: int,
        include_attention: bool = False,
        attention_top_n: int = 5,
    ) -> dict[str, Any]:
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

        attn_weights = None
        if include_attention:
            logits, attn_weights = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_last_attn=True,
            )
        else:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        next_logits = logits[0, -1].clone()
        next_logits[0] = float("-inf")  # never recommend PAD
        # Avoid echoing artists that are already in the user's provided history.
        seen_token_ids = sorted(set(known_tokens))
        next_logits[seen_token_ids] = float("-inf")
        k = min(top_k, next_logits.shape[-1])
        top_scores, top_tokens = torch.topk(next_logits, k=k)

        probs = torch.softmax(next_logits, dim=-1)
        predictions: list[dict[str, Any]] = []
        for token, score in zip(top_tokens.tolist(), top_scores.tolist(), strict=False):
            artist_id = self.token_to_artist.get(int(token), f"<unk-token-{token}>")
            predictions.append(
                {
                    "artist_id": artist_id,
                    "artist_name": self._artist_name(artist_id),
                    "artist_image_url": self._artist_image_url(artist_id, lazy=True),
                    "token_id": int(token),
                    "logit": float(score),
                    "prob": float(probs[token].item()),
                }
            )

        used_artist_ids = [self.token_to_artist[t] for t in known_tokens if t in self.token_to_artist]
        used_artist_names = [self._artist_name(artist_id) for artist_id in used_artist_ids]
        used_artist_images = [self._artist_image_url(artist_id, lazy=True) for artist_id in used_artist_ids]
        response = {
            "history_used_artist_ids": used_artist_ids,
            "history_used_artist_names": used_artist_names,
            "history_used_artist_images": used_artist_images,
            "unknown_artist_ids": unknown_artist_ids,
            "top_k": k,
            "predictions": predictions,
        }
        if include_attention and attn_weights is not None:
            # attn_weights: [batch, heads, target_len, source_len]
            attn_last_query = attn_weights[0, :, -1, : len(used_artist_ids)]
            mean_weights = attn_last_query.mean(dim=0)

            per_head: list[dict[str, Any]] = []
            for head_idx in range(attn_last_query.shape[0]):
                w = attn_last_query[head_idx]
                topn = min(attention_top_n, w.shape[0])
                top_vals, top_pos = torch.topk(w, k=topn)
                contributors: list[dict[str, Any]] = []
                for pos, weight in zip(top_pos.tolist(), top_vals.tolist(), strict=False):
                    contributors.append(
                        {
                            "position": int(pos),
                            "artist_id": used_artist_ids[pos],
                            "artist_name": used_artist_names[pos],
                            "weight": float(weight),
                        }
                    )
                per_head.append(
                    {
                        "head_index": int(head_idx),
                        "weights": [float(x) for x in w.tolist()],
                        "top_contributors": contributors,
                    }
                )

            response["attention"] = {
                "num_heads": int(attn_last_query.shape[0]),
                "seq_len": int(attn_last_query.shape[1]),
                "history_artist_ids": used_artist_ids,
                "history_artist_names": used_artist_names,
                "mean_weights": [float(x) for x in mean_weights.tolist()],
                "per_head": per_head,
            }
        return response

    def search_artists(self, query: str, limit: int) -> list[dict[str, Any]]:
        q = query.strip().lower()
        if not q:
            candidates = self.artist_ids[:limit]
        else:
            starts = [
                a
                for a in self.artist_ids
                if a.lower().startswith(q) or self._artist_name(a).lower().startswith(q)
            ]
            contains = [
                a
                for a in self.artist_ids
                if (
                    q in a.lower() or q in self._artist_name(a).lower()
                ) and (
                    not a.lower().startswith(q) and not self._artist_name(a).lower().startswith(q)
                )
            ]
            candidates = (starts + contains)[:limit]

        return [
            {
                "artist_id": a,
                "artist_name": self._artist_name(a),
                "artist_image_url": self._artist_image_url(a, lazy=False),
                "token_id": int(self.vocab[a]),
            }
            for a in candidates
        ]


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
        return server.predict(
            history_artist_ids=req.history_artist_ids,
            top_k=req.top_k,
            include_attention=req.include_attention,
            attention_top_n=req.attention_top_n,
        )

    @app.get("/artist_media")
    def artist_media(artist_id: str = Query(..., description="Artist id")) -> dict[str, Any]:
        if artist_id not in server.vocab:
            raise HTTPException(status_code=404, detail="artist_id not found")
        return {
            "artist_id": artist_id,
            "artist_name": server._artist_name(artist_id),
            "artist_image_url": server._artist_image_url(artist_id, lazy=True),
        }

    @app.middleware("http")
    async def disable_static_caching(request, call_next):
        response: Response = await call_next(request)
        path = request.url.path
        if path in ("/", "/index.html") or path.endswith(".js") or path.endswith(".css"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

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
