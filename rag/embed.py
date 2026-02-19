from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Returns float32 embeddings, L2-normalized for cosine similarity with FAISS IP index.
        """
        emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        emb = emb.astype("float32")

        # L2 normalize so inner product == cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        return emb
