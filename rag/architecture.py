from __future__ import annotations

from pathlib import Path
from typing import List

from ingest.filter import list_source_files
from ingest.loader import load_documents
from ingest.chunk import chunk_documents, CodeChunk


# Generic keywords to find "core" files across arbitrary repos
CORE_KEYWORDS = [
    "app", "application", "main", "server",
    "routing", "router", "routes",
    "api", "endpoint", "handlers", "controllers",
    "service", "usecase", "domain",
    "dependencies", "di", "inject",
    "middleware", "auth", "security",
    "openapi", "schema",
    "config", "settings",
    "exceptions", "errors",
]

# Repo noise / non-core areas (repo-agnostic)
EXCLUDE_DIRS = {
    "docs", "doc", "docs_src", "examples", "example", "tutorial", "tutorials",
    ".github", ".git", "tests", "test", "__pycache__",
    "site-packages", "node_modules", "dist", "build", "coverage", ".venv", "venv",
}

EXCLUDE_FILES = {
    "LICENSE", "NOTICE", "CHANGELOG.md", "HISTORY.md", "CONTRIBUTING.md",
}


def detect_python_packages(repo_root: Path, files: List[Path]) -> set[str]:
    """
    Return package directories (relative) that contain __init__.py.
    Example: {"fastapi", "app", "src/my_pkg"}
    """
    packages: set[str] = set()
    for f in files:
        rel = f.relative_to(repo_root)
        if rel.name == "__init__.py":
            packages.add(str(rel.parent))
    return packages


def score_file(rel_path: str, packages: set[str] | None = None) -> int:
    """
    Score a file path by likelihood of being "core architecture".
    Higher score => more likely to contain meaningful modules/entrypoints.
    """
    p = Path(rel_path)
    parts = [x.lower() for x in p.parts]

    # Exclude obvious noise dirs/files
    if any(part in EXCLUDE_DIRS for part in parts):
        return -999
    if p.name in EXCLUDE_FILES:
        return -999

    score = 0
    ext = p.suffix.lower()

    # Prefer python source
    if ext == ".py":
        score += 5

    # Prefer src/app roots (common monorepo layout)
    if parts and parts[0] in {"src", "app"}:
        score += 3

    # Boost if inside a detected python package directory
    if packages:
        rel_str = str(p)
        for pkg in packages:
            if rel_str == pkg or rel_str.startswith(pkg + "/"):
                score += 8
                break

    name = p.stem.lower()

    # Entry points (reduced weight to avoid docs examples dominating)
    if p.name.lower() in {"main.py", "app.py", "__main__.py"}:
        score += 6

    # Keyword hits (file name or path segments)
    for kw in CORE_KEYWORDS:
        if kw in name or any(kw in part for part in parts):
            score += 2

    return score


def pick_hotspot_chunks(
    repo_root: Path,
    chunks: List[CodeChunk],
    packages: set[str],
    max_chunks: int = 24,
) -> List[CodeChunk]:
    """
    Pick architecture-relevant chunks across the repo in a repo-agnostic way.

    Strategy:
    1) Group chunks by file
    2) Rank files using score_file(...)
    3) For top files, pick:
       - 1 FILE_HEADER chunk (module overview) if present
       - up to 2 symbol chunks (classes/functions/methods), skipping generic BLOCK chunks
    Ensures coverage across multiple files.
    """
    by_file: dict[str, List[CodeChunk]] = {}

    for c in chunks:
        rel = c.rel_path
        p = Path(rel)
        parts = [x.lower() for x in p.parts]

        if any(part in EXCLUDE_DIRS for part in parts):
            continue
        if p.name in EXCLUDE_FILES:
            continue

        by_file.setdefault(rel, []).append(c)

    ranked_files = sorted(by_file.keys(), key=lambda rp: score_file(rp, packages), reverse=True)

    picked: List[CodeChunk] = []
    for rel in ranked_files:
        if len(picked) >= max_chunks:
            break

        file_chunks = by_file[rel]

        # Prefer module overview
        headers = [c for c in file_chunks if c.symbol == "FILE_HEADER"]
        if headers:
            picked.append(headers[0])

        # Prefer real symbols (classes/functions/methods)
        symbols = [c for c in file_chunks if c.symbol not in {"FILE_HEADER", "BLOCK"}]
        picked.extend(symbols[:2])

    return picked[:max_chunks]


def explain_architecture(llm_chat, repo_root: Path, debug_print_sources: bool = True) -> str:
    """
    Architecture overview with reliable citations via source IDs [S1], [S2], ...
    Works across arbitrary repos (not FastAPI-specific).
    """
    repo_root = repo_root.resolve()

    files, _ = list_source_files(repo_root)
    packages = detect_python_packages(repo_root, files)

    docs = load_documents(repo_root, files)
    chunks = chunk_documents(docs)

    hotspots = pick_hotspot_chunks(repo_root, chunks, packages, max_chunks=18)

    # Build a strict source catalog with IDs
    catalog_blocks: List[str] = []
    for i, c in enumerate(hotspots, start=1):
        sid = f"S{i}"
        ref = f"{c.rel_path}:{c.start_line}-{c.end_line}"
        sym = c.symbol
        text = c.text.strip()
        if len(text) > 1200:
            text = text[:1200] + "\n... (truncated)"
        catalog_blocks.append(f"[{sid}] {ref} | {sym}\n{text}\n")

    context = "\n".join(catalog_blocks)

    if debug_print_sources:
        print("\n[ARCH SOURCES CATALOG]")
        for i, c in enumerate(hotspots, start=1):
            print(f"- [S{i}] {c.rel_path}:{c.start_line}-{c.end_line} | {c.symbol}")

    system = (
    "You are a codebase architect.\n"
    "Use ONLY the SOURCES below.\n\n"
    "You MUST cite using the source IDs exactly like [S1], [S2] at the end of every bullet.\n"
    "Never use the phrase 'not shown in sources'. If it isn't in sources, omit it.\n\n"
    "You MUST output EXACTLY this structure for each module (repeat 5-8 times):\n"
    "Module: <exact file path from sources>\n"
    "- Responsibility: <one sentence> [S#]\n"
    "- Key symbols: <comma-separated symbols> [S#]\n\n"
    "Rules:\n"
    "- 'Module' must match a file path that appears in SOURCES (e.g., fastapi/routing.py).\n"
    "- Do not invent symbols.\n"
    )

    user = (
        "Create an architecture overview of this repository.\n"
        "Prefer these areas when applicable: entrypoints, app initialization, routing, services, "
        "dependency injection, config, schema/OpenAPI, exception handling.\n\n"
        f"SOURCES:\n{context}"
    )

    return llm_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
