from __future__ import annotations

import json
import re
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


def safe_parse_json(raw: str) -> dict:
    """
    Parse JSON even if the model adds extra text.
    Tries:
    1) direct json.loads
    2) extract first {...} block
    """
    if raw is None:
        raise ValueError("LLM returned None")

    s = raw.strip()
    if not s:
        raise ValueError("LLM returned empty string")

    # 1) direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2) extract first JSON object block
    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON from LLM output")

def sanitize_arch_json(parsed: dict, catalog: list[dict], target_modules: int = 6) -> dict:
    """
    Ensure:
    - module paths exist in catalog
    - citations belong to the same file path
    - key_symbols align with the same file path
    - always returns exactly `target_modules` modules (backfilled deterministically)
    """
    # Map source id -> (path, symbol)
    sid_to_path: dict[str, str] = {}
    sid_to_symbol: dict[str, str] = {}
    path_to_sids: dict[str, list[str]] = {}
    path_to_symbols: dict[str, list[str]] = {}

    for item in catalog:
        sid = item["id"]
        path = item["ref"].split(":")[0]  # file path
        sym = item["symbol"]

        sid_to_path[sid] = path
        sid_to_symbol[sid] = sym
        path_to_sids.setdefault(path, []).append(sid)
        path_to_symbols.setdefault(path, []).append(sym)

    valid_paths = set(path_to_sids.keys())

    modules = parsed.get("modules", [])
    if not isinstance(modules, list):
        modules = []

    cleaned: list[dict] = []
    used_paths: set[str] = set()

    for m in modules:
        if not isinstance(m, dict):
            continue

        path = (m.get("path") or "").lstrip("/")  # normalize "/fastapi/x.py" -> "fastapi/x.py"
        if not path or path not in valid_paths:
            continue
        if path in used_paths:
            continue

        # Keep only citations that actually belong to this file
        citations = m.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        citations = [c for c in citations if c in sid_to_path and sid_to_path[c] == path]

        # If model gave none/incorrect, pick the first catalog source for that path
        if not citations:
            citations = [path_to_sids[path][0]]

        # Key symbols: keep only those appearing in this path’s catalog symbols
        key_symbols = m.get("key_symbols", [])
        if not isinstance(key_symbols, list):
            key_symbols = []
        allowed_syms = set(path_to_symbols.get(path, []))
        key_symbols = [s for s in key_symbols if s in allowed_syms]

        # If empty, derive from the first citation’s symbol
        if not key_symbols:
            sym = sid_to_symbol[citations[0]]
            # Replace FILE_HEADER with file stem (better UI)
            if sym == "FILE_HEADER":
                sym = Path(path).stem
            key_symbols = [sym]

        responsibility = (m.get("responsibility") or "").strip()
        if not responsibility:
            responsibility = f"Core module containing {key_symbols[0]}."

        cleaned.append(
            {
                "path": path,
                "responsibility": responsibility,
                "key_symbols": key_symbols[:3],
                "citations": citations[:3],
            }
        )
        used_paths.add(path)

        if len(cleaned) >= target_modules:
            break

    # Backfill deterministically from catalog if we have < target_modules
    if len(cleaned) < target_modules:
        for path in sorted(valid_paths):
            if path in used_paths:
                continue
            sid = path_to_sids[path][0]
            sym = sid_to_symbol[sid]
            if sym == "FILE_HEADER":
                sym = Path(path).stem

            cleaned.append(
                {
                    "path": path,
                    "responsibility": f"Core module containing {sym}.",
                    "key_symbols": [sym],
                    "citations": [sid],
                }
            )
            used_paths.add(path)
            if len(cleaned) >= target_modules:
                break

    return {"modules": cleaned[:target_modules]}


def explain_architecture_with_sources(llm_chat, repo_root: Path, max_chunks: int = 18):
    """
    Returns (arch_json_dict, catalog_items)
    catalog_items: list of dicts {id, ref, symbol, text}
    """
    repo_root = repo_root.resolve()

    files, _ = list_source_files(repo_root)
    packages = detect_python_packages(repo_root, files)

    docs = load_documents(repo_root, files)
    chunks = chunk_documents(docs)

    hotspots = pick_hotspot_chunks(repo_root, chunks, packages, max_chunks=max_chunks)

    # Build catalog
    catalog: list[dict] = []
    for i, c in enumerate(hotspots, start=1):
        sid = f"S{i}"
        ref = f"{c.rel_path}:{c.start_line}-{c.end_line}"
        sym = c.symbol
        text = c.text.strip()
        if len(text) > 2000:
            text = text[:2000] + "\n... (truncated)"
        catalog.append({"id": sid, "ref": ref, "symbol": sym, "text": text})

    # ✅ Build compact SOURCES list (metadata only)
    catalog_lines = [
        f"{item['id']} | {item['ref']} | {item['symbol']}"
        for item in catalog
    ]
    context = "\n".join(catalog_lines)

    system = (
        "You are a codebase architect.\n"
        "Use ONLY the SOURCES list.\n"
        "Return ONLY JSON.\n"
        "You MUST return exactly 6 modules.\n"
        "Do not return an empty list.\n"
        "Schema:\n"
        '{"modules":[{"path":"...","responsibility":"...","key_symbols":["..."],"citations":["S1"]}]}\n'
    )

    user = (
        "Pick exactly 6 core modules that best represent the repo architecture.\n"
        "Rules:\n"
        "- Each module must come from a DIFFERENT file path.\n"
        "- path must match the file path in SOURCES.\n"
        "- citations must reference the source IDs used (e.g. S3).\n"
        "- key_symbols must be the symbol names from SOURCES (e.g. FastAPI, request_response).\n"
        "- responsibility: one short sentence.\n\n"
        f"SOURCES:\n{context}"
    )

    def _ask() -> str:
        return llm_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            format="json",
        )

    raw1 = _ask()
    print("\n[ARCH RAW JSON - first 400 chars]\n", (raw1 or "")[:400])

    try:
        parsed = safe_parse_json(raw1)
    except Exception as e:
        print("\n[ARCH JSON PARSE ERROR 1]", repr(e))

        raw2 = llm_chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user + "\n\nREMINDER: Return ONLY valid JSON. Start with '{' and end with '}'. No extra text."},
            ],
            format="json",
        )
        print("\n[ARCH RAW JSON RETRY - first 400 chars]\n", (raw2 or "")[:400])

        try:
            parsed = safe_parse_json(raw2)
        except Exception as e2:
            print("\n[ARCH JSON PARSE ERROR 2]", repr(e2))
            parsed = {"modules": []}

    # Guarantee dict
    if not isinstance(parsed, dict):
        parsed = {"modules": []}

    # Guarantee modules list
    if "modules" not in parsed or not isinstance(parsed["modules"], list):
        parsed["modules"] = []

    # ✅ Deterministic fallback (never empty)
    if len(parsed["modules"]) == 0:
        seen = set()
        fallback = []
        for item in catalog:
            item_path = item["ref"].split(":")[0]
            if item_path in seen:
                continue

            sym = item["symbol"]
            if sym == "FILE_HEADER":
                sym = Path(item_path).stem  # better than "FILE_HEADER"

            seen.add(item_path)
            fallback.append({
                "path": item_path,
                "responsibility": f"Core module containing {sym}.",
                "key_symbols": [sym],
                "citations": [item["id"]],
            })

            if len(fallback) == 6:
                break

        parsed["modules"] = fallback
    
    parsed = sanitize_arch_json(parsed, catalog, target_modules=6)


    return parsed, catalog
