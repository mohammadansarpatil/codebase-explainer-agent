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
    - If .py => AST chunk into functions/classes
    - Else => skip for now (we'll add heuristic chunking later)
    """
    all_chunks: List[CodeChunk] = []
    for d in docs:
        if d.rel_path.endswith(".py"):
            all_chunks.extend(chunk_python_document(d))
    return all_chunks
