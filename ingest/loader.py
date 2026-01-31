from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    rel_path: str
    abs_path: str
    text: str


def read_text_file(path: Path) -> str:
    """
    Read a file as text with safe fallbacks.
    - Try UTF-8 first
    - If it fails, try latin-1
    - If it still fails, read as UTF-8 with replacement characters
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")


def load_documents(repo_path: Path, file_paths: List[Path]) -> List[Document]:
    """
    Convert a list of absolute file paths into Document objects with:
    - rel_path: path relative to repo root
    - abs_path: full path for debugging
    - text: file content
    """
    repo_path = repo_path.resolve()
    docs: List[Document] = []

    for p in file_paths:
        p = p.resolve()
        rel = str(p.relative_to(repo_path))
        text = read_text_file(p)

        docs.append(
            Document(
                rel_path=rel,
                abs_path=str(p),
                text=text,
            )
        )

    return docs
