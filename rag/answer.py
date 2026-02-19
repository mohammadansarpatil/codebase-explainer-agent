from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from ingest.loader import read_text_file
from rag.retrieve import retrieve


@dataclass
class SourceSnippet:
    rel_path: str
    start_line: int
    end_line: int
    symbol: str
    text: str
    score: float


def _slice_lines(text: str, start_line: int, end_line: int) -> str:
    lines = text.splitlines()
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return "\n".join(lines[start_idx:end_idx])


def build_sources(repo_root: Path, hits: List[dict], max_sources: int = 6, min_code_sources: int = 3) -> List[SourceSnippet]:
    """
    Convert retrieval hits into real snippets.
    Ensures we include at least min_code_sources from code files when available.
    """
    repo_root = repo_root.resolve()

    # Split hits into code-like and others
    code_hits = []
    other_hits = []
    for h in hits:
        rel = h["rel_path"]
        ext = Path(rel).suffix.lower()
        if ext in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".cs"}:
            code_hits.append(h)
        else:
            other_hits.append(h)

    chosen = []
    chosen.extend(code_hits[:min_code_sources])

    # Fill remaining slots with best remaining hits (avoid duplicates)
    seen = {f"{h['rel_path']}:{h['start_line']}-{h['end_line']}" for h in chosen}
    for h in code_hits[min_code_sources:] + other_hits:
        key = f"{h['rel_path']}:{h['start_line']}-{h['end_line']}"
        if key in seen:
            continue
        chosen.append(h)
        seen.add(key)
        if len(chosen) >= max_sources:
            break

    sources: List[SourceSnippet] = []
    for h in chosen:
        rel_path = h["rel_path"]
        abs_path = repo_root / rel_path
        file_text = read_text_file(abs_path)
        snippet = _slice_lines(file_text, h["start_line"], h["end_line"])

        sources.append(
            SourceSnippet(
                rel_path=rel_path,
                start_line=int(h["start_line"]),
                end_line=int(h["end_line"]),
                symbol=h["symbol"],
                text=snippet,
                score=float(h.get("reranked_score", h["score"])),
            )
        )

    return sources

def format_context(sources: List[SourceSnippet], max_chars_per_source: int = 1800) -> str:
    blocks = []
    for s in sources:
        text = s.text.strip()
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source] + "\n... (truncated)"
        blocks.append(
            f"[SOURCE] {s.rel_path}:{s.start_line}-{s.end_line} | {s.symbol}\n{text}\n"
        )
    return "\n".join(blocks)



def answer_question(
    *,
    llm_chat,  # function: (messages)->str
    repo_root: Path,
    index_dir: Path,
    embedder,
    question: str,
    top_k: int = 10,
) -> tuple[str, list[SourceSnippet]]:
    """
    Full RAG: retrieve -> build context -> ask LLM -> return answer.
    """
    qvec = embedder.encode([question])[0]
    hits = retrieve(index_dir, qvec, k=max(top_k, 30))

    print("\n[SOURCES USED FOR ANSWER]")
    for h in hits[:6]:
        print("-", f"{h['rel_path']}:{h['start_line']}-{h['end_line']}", "|", h["symbol"], "|", round(h.get("reranked_score", h["score"]), 4))


    sources = build_sources(repo_root, hits, max_sources=6, min_code_sources=3)
    context = format_context(sources)

    system = (
        "You are a codebase explainer. You MUST use ONLY the provided SOURCES.\n"
        "Do not use outside knowledge.\n\n"
        "Output format:\n"
        "1) Overview (2-4 lines)\n"
        "2) Step-by-step flow (4-8 bullets)\n"
        "3) Key files & functions referenced (3-6 bullets)\n\n"
        "Rules:\n"
        "- Every bullet MUST end with an inline citation in this exact format: (path:start-end)\n"
        "- Do NOT use numbered references like (1), (2), (3).\n"
        "- Do NOT mention any file/function not present in SOURCES.\n"
        "- If something is not in sources, say 'Not in sources.'\n"
    )

    user = (
        f"Question: {question}\n\n"
        f"SOURCES:\n{context}\n\n"
        "Now answer using the rules."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    answer = llm_chat(messages)
    return answer, sources