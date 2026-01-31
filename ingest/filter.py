from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple


@dataclass
class FilterStats:
    total_files_seen: int
    kept_files: int
    skipped_by_dir: int
    skipped_by_ext: int
    skipped_by_size: int
    skipped_non_text: int


# Directories to skip entirely (common across many repos)
DEFAULT_SKIP_DIRS: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    "dist",
    "build",
    "target",
    "coverage",
    ".idea",
    ".vscode",
}


# Extensions usually not useful for code understanding / RAG
DEFAULT_SKIP_EXTS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
    ".mp4", ".mov", ".avi", ".mp3", ".wav",
    ".exe", ".dll", ".so", ".dylib",
    ".bin", ".dat",
    ".woff", ".woff2", ".ttf", ".otf",
}


# Extensions we generally *do* want (helpful for architecture + setup)
DEFAULT_ALLOW_EXTS: Set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".html", ".css", ".scss",
    ".md", ".txt", ".rst",
    ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh",
    ".sql",
    "Dockerfile",  # filename-based
    "Makefile",
}


def _is_probably_text_file(path: Path, max_bytes: int = 4096) -> bool:
    """
    Quick heuristic: read first N bytes; if it contains NUL bytes it's likely binary.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(max_bytes)
        return b"\x00" not in chunk
    except Exception:
        return False


def list_source_files(
    repo_path: Path,
    *,
    max_file_size_bytes: int = 1_500_000,
    skip_dirs: Set[str] = DEFAULT_SKIP_DIRS,
    skip_exts: Set[str] = DEFAULT_SKIP_EXTS,
    allow_exts: Set[str] = DEFAULT_ALLOW_EXTS,
) -> Tuple[List[Path], FilterStats]:
    """
    Walk repo_path and return (kept_files, stats) after applying filters:
    - skip directories in skip_dirs
    - skip by extension in skip_exts
    - allow by extension in allow_exts (or special filenames)
    - skip files larger than max_file_size_bytes
    - skip likely-binary files
    """
    repo_path = repo_path.resolve()

    kept: List[Path] = []

    total_seen = 0
    skipped_by_dir = 0
    skipped_by_ext = 0
    skipped_by_size = 0
    skipped_non_text = 0

    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)

        # Modify dirs in-place so os.walk doesn't descend into skipped dirs
        new_dirs = []
        for d in dirs:
            if d in skip_dirs:
                skipped_by_dir += 1
            else:
                new_dirs.append(d)
        dirs[:] = new_dirs

        for filename in files:
            total_seen += 1
            file_path = root_path / filename

            # Size filter
            try:
                if file_path.stat().st_size > max_file_size_bytes:
                    skipped_by_size += 1
                    continue
            except Exception:
                skipped_by_size += 1
                continue

            # Allowlist logic:
            # - If filename is Dockerfile/Makefile -> keep
            # - Else check extension
            if filename in allow_exts:
                pass
            else:
                ext = file_path.suffix.lower()
                if ext in skip_exts:
                    skipped_by_ext += 1
                    continue
                if ext and (ext not in allow_exts):
                    # Unknown extension: skip to keep week-1 scope tight
                    skipped_by_ext += 1
                    continue

            # Binary sniff
            if not _is_probably_text_file(file_path):
                skipped_non_text += 1
                continue

            kept.append(file_path)

    stats = FilterStats(
        total_files_seen=total_seen,
        kept_files=len(kept),
        skipped_by_dir=skipped_by_dir,
        skipped_by_ext=skipped_by_ext,
        skipped_by_size=skipped_by_size,
        skipped_non_text=skipped_non_text,
    )

    return kept, stats
