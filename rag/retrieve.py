from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np

from rag.index import load_index


DOC_PENALTY_FILES = {
    "README.md", "README.rst", "LICENSE", "NOTICE", "HISTORY.md", "CHANGELOG.md",
}


def _is_doc_like(rel_path: str) -> bool:
    name = Path(rel_path).name
    if name in DOC_PENALTY_FILES:
        return True
    ext = Path(rel_path).suffix.lower()
    return ext in {".md", ".rst", ".txt"}


def _is_code_like(rel_path: str) -> bool:
    ext = Path(rel_path).suffix.lower()
    return ext in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".cs"}


def retrieve(index_dir: Path, query_vec: np.ndarray, k: int = 12) -> List[dict]:
    """
    Returns top-k metadata rows with score, with:
    - de-duplication by (rel_path, start_line, end_line)
    - light re-ranking to prefer code for code-ish queries
    """
    index, meta = load_index(index_dir)

    q = np.asarray([query_vec], dtype="float32")
    scores, ids = index.search(q, k)

    # Build raw rows
    raw: List[dict] = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        row = dict(meta[i])
        row["score"] = float(score)
        raw.append(row)

    # De-duplicate: keep best score for same span
    best_by_span: Dict[str, dict] = {}
    for r in raw:
        key = f"{r['rel_path']}:{r['start_line']}-{r['end_line']}"
        if key not in best_by_span or r["score"] > best_by_span[key]["score"]:
            best_by_span[key] = r

    results = list(best_by_span.values())

    # Light re-rank:
    # If query seems "code-ish", penalize doc-like files a bit.
    query_hint = ""  # we don't have the text here, so keep it simple always-on
    for r in results:
        if _is_doc_like(r["rel_path"]) and not _is_code_like(r["rel_path"]):
            r["reranked_score"] = r["score"] * 0.85
        else:
            r["reranked_score"] = r["score"]

    results.sort(key=lambda x: x["reranked_score"], reverse=True)
    return results
