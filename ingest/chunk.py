from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ingest.loader import Document


@dataclass
class CodeChunk:
    chunk_id: str
    rel_path: str
    symbol: str
    start_line: int
    end_line: int
    text: str


class _SymbolVisitor(ast.NodeVisitor):
    """
    Visits a Python AST and extracts:
    - top-level functions
    - classes
    - methods inside classes
    """

    def __init__(self) -> None:
        self.chunks: List[tuple[str, int, int]] = []
        self._class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        class_name = node.name
        self.chunks.append((class_name, node.lineno, getattr(node, "end_lineno", node.lineno)))
        self._class_stack.append(class_name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        fn_name = node.name
        if self._class_stack:
            symbol = f"{self._class_stack[-1]}.{fn_name}"
        else:
            symbol = fn_name

        self.chunks.append((symbol, node.lineno, getattr(node, "end_lineno", node.lineno)))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # treat async same as normal functions
        fn_name = node.name
        if self._class_stack:
            symbol = f"{self._class_stack[-1]}.{fn_name}"
        else:
            symbol = fn_name

        self.chunks.append((symbol, node.lineno, getattr(node, "end_lineno", node.lineno)))
        self.generic_visit(node)


def _slice_lines(text: str, start_line: int, end_line: int) -> str:
    """
    Return text from start_line..end_line (1-based inclusive).
    """
    lines = text.splitlines()
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return "\n".join(lines[start_idx:end_idx])

def make_file_header_chunk(rel_path: str, text: str, max_lines: int = 80) -> CodeChunk:
    """
    Create a small chunk for the top of a file (imports, constants, docstring, config headers).
    """
    lines = text.splitlines()
    end_line = min(len(lines), max_lines)
    header_text = "\n".join(lines[:end_line]).strip()

    chunk_id = f"{rel_path}::FILE_HEADER::1-{end_line}::0"
    return CodeChunk(
        chunk_id=chunk_id,
        rel_path=rel_path,
        symbol="FILE_HEADER",
        start_line=1,
        end_line=end_line,
        text=header_text,
    )


def chunk_text_by_max_lines(rel_path: str, text: str, lines_per_chunk: int = 120) -> List[CodeChunk]:
    """
    Generic fallback chunker: split text into fixed-size line windows.
    Good enough for configs/docs in week-1.
    """
    lines = text.splitlines()
    chunks: List[CodeChunk] = []
    if not lines:
        return chunks

    idx = 0
    start = 1
    while start <= len(lines):
        end = min(start + lines_per_chunk - 1, len(lines))
        chunk_text = "\n".join(lines[start - 1:end]).strip()
        chunk_id = f"{rel_path}::BLOCK::{start}-{end}::{idx}"

        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                rel_path=rel_path,
                symbol="BLOCK",
                start_line=start,
                end_line=end,
                text=chunk_text,
            )
        )

        idx += 1
        start = end + 1

    return chunks


def chunk_python_document(doc: Document) -> List[CodeChunk]:
    """
    Chunk a Python file Document into function/class-level chunks using AST.
    """
    try:
        tree = ast.parse(doc.text)
    except SyntaxError:
        # Some files may be incompatible (rare); skip chunking and return empty
        return []

    visitor = _SymbolVisitor()
    visitor.visit(tree)

    chunks: List[CodeChunk] = []
    for idx, (symbol, start, end) in enumerate(visitor.chunks):
        chunk_text = _slice_lines(doc.text, start, end)
        chunk_id = f"{doc.rel_path}::{symbol}::{start}-{end}::{idx}"

        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                rel_path=doc.rel_path,
                symbol=symbol,
                start_line=start,
                end_line=end,
                text=chunk_text,
            )
        )

    return chunks


def chunk_documents(docs: List[Document]) -> List[CodeChunk]:
    """
    Chunk all documents:
    - Always add a FILE_HEADER chunk (if file has content)
    - If .py => AST chunks (functions/classes/methods)
    - Else => fallback line-window chunking (BLOCK)
    """
    all_chunks: List[CodeChunk] = []

    for d in docs:
        # Header chunk for every file
        header = make_file_header_chunk(d.rel_path, d.text, max_lines=80)
        if header.text:
            all_chunks.append(header)

        # Python: AST chunking
        if d.rel_path.endswith(".py"):
            all_chunks.extend(chunk_python_document(d))
        else:
            # Non-python: simple fixed window chunks
            all_chunks.extend(chunk_text_by_max_lines(d.rel_path, d.text, lines_per_chunk=120))

    return all_chunks
