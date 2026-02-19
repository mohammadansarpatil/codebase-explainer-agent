from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict


@dataclass
class RepoMap:
    repo_root: str
    total_files: int
    top_level_dirs: List[str]
    important_files: List[str]
    likely_entrypoints: List[str]
    ext_counts: Dict[str, int]


IMPORTANT_FILENAMES = {
    "README.md",
    "README.rst",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "package.json",
    "tsconfig.json",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
    ".env.example",
}


ENTRYPOINT_HINTS = {
    "main.py",
    "app.py",
    "server.py",
    "wsgi.py",
    "asgi.py",
    "manage.py",
    "index.js",
    "server.js",
    "main.ts",
    "index.ts",
}


def build_repo_map(repo_path: Path, kept_files: List[Path]) -> RepoMap:
    repo_path = repo_path.resolve()

    top_dirs = set()
    important = []
    entrypoints = []
    ext_counter = Counter()

    for p in kept_files:
        p = p.resolve()
        rel = p.relative_to(repo_path)
        parts = rel.parts

        if len(parts) >= 2:
            top_dirs.add(parts[0])

        name = rel.name
        if name in IMPORTANT_FILENAMES:
            important.append(str(rel))

        if name in ENTRYPOINT_HINTS:
            entrypoints.append(str(rel))

        ext = p.suffix.lower() if p.suffix else name  # for Dockerfile/Makefile no suffix
        ext_counter[ext] += 1

    # Sort for stable output
    top_level_dirs = sorted(list(top_dirs))
    important_files = sorted(important)
    likely_entrypoints = sorted(entrypoints)
    ext_counts = dict(ext_counter.most_common())

    return RepoMap(
        repo_root=str(repo_path),
        total_files=len(kept_files),
        top_level_dirs=top_level_dirs,
        important_files=important_files,
        likely_entrypoints=likely_entrypoints,
        ext_counts=ext_counts,
    )
