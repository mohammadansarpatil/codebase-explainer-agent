from __future__ import annotations

from pathlib import Path
import hashlib

from ingest.clone import clone_repo
from ingest.filter import list_source_files
from ingest.loader import load_documents
from ingest.chunk import chunk_documents

from rag.embed import Embedder
from rag.index import build_faiss_index, persist_index


def build_index_for_repo(repo_url: str, update: bool = False) -> tuple[Path, Path]:
    """
    Returns (repo_root_path, index_dir).
    Builds index only if it doesn't exist.
    """
    clone_result = clone_repo(repo_url, update_if_exists=update)
    repo_root = Path(clone_result.local_path)

    # Files -> docs -> chunks
    files, _stats = list_source_files(repo_root)
    docs = load_documents(repo_root, files)
    chunks = chunk_documents(docs)

    # Embeddings + FAISS
    embedder = Embedder()
    embeddings = embedder.encode([c.text for c in chunks])

    index = build_faiss_index(embeddings)

    repo_id = hashlib.sha1(repo_url.encode("utf-8")).hexdigest()[:12]
    index_dir = Path("data/indexes") / repo_id
    persist_index(index_dir, index, chunks)

    return repo_root, index_dir
