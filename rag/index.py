from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from ingest.chunk import CodeChunk


def save_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    embeddings: (n, d) float32 L2-normalized.
    Use inner product index (cosine if normalized).
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def persist_index(index_dir: Path, index: faiss.Index, chunks: List[CodeChunk]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS
    faiss.write_index(index, str(index_dir / "index.faiss"))

    # Save metadata
    meta_rows = []
    for c in chunks:
        meta_rows.append({
            "chunk_id": c.chunk_id,
            "rel_path": c.rel_path,
            "symbol": c.symbol,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "text_preview": c.text[:300],
        })
    save_jsonl(index_dir / "meta.jsonl", meta_rows)


def load_index(index_dir: Path) -> Tuple[faiss.Index, List[dict]]:
    index = faiss.read_index(str(index_dir / "index.faiss"))

    meta = []
    with open(index_dir / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta
