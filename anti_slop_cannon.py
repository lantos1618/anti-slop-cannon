#!/usr/bin/env python3
"""Semantic clustering map for spotting repeated codebase files."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_EXCLUDE_DIRS = {
    ".anti-slop-cache",
    ".anti-slop-out",
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "__pycache__",
}

DEFAULT_EXCLUDE_FILES = {
    "anti_slop_report.json",
    "anti_slop_map.html",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}

DEFAULT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".clj",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".lua",
    ".md",
    ".mdx",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#9333ea",
    "#d97706",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#16a34a",
    "#7c2d12",
]


@dataclass
class FileDoc:
    path: str
    size: int
    sha256: str
    extension: str
    text: str
    imports: list[str]
    calls: list[str]


@dataclass
class SemanticItem:
    id: str
    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    size: int
    sha256: str
    normalized_hash: str
    extension: str
    text: str
    imports: list[str]
    calls: list[str]


@dataclass
class SimilarPair:
    a: str
    b: str
    similarity: float
    relation: str = "semantic"
    evidence: str = ""


@dataclass
class SlopExample:
    id: str
    name: str
    source: str
    size: int
    sha256: str
    text: str


@dataclass
class SlopMatch:
    example_id: str
    example_name: str
    item_id: str
    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    similarity: float
    relation: str
    evidence: str


@dataclass
class Cluster:
    id: int
    label: str
    items: list[str]
    files: list[str]
    density: float
    max_similarity: float
    semantic_edges: int
    exact_edges: int
    near_edges: int
    shared_imports: list[str]
    shared_calls: list[str]
    recommendation: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed each codebase file and visualize semantic overlap clusters."
    )
    parser.add_argument("root", nargs="?", default=".", help="Codebase root to scan.")
    parser.add_argument(
        "--provider",
        choices=("google", "openai", "openrouter", "sentence-transformers", "hash"),
        default="google",
        help="Embedding backend. Use hash only for offline smoke tests.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Embedding model. Defaults to gemini-embedding-2 for Google, "
            "text-embedding-3-large for OpenAI, openai/text-embedding-3-small "
            "for OpenRouter, Qwen/Qwen3-Embedding-8B for sentence-transformers, "
            "or hash-ngram."
        ),
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=None,
        help="Embedding dimensions to request/use when the provider supports it.",
    )
    parser.add_argument(
        "--granularity",
        choices=("file", "symbol", "both"),
        default="symbol",
        help="Embed whole files, extracted symbols/chunks, or both.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Cosine similarity cutoff for cluster edges.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.86,
        help="Token-overlap cutoff for near-duplicate edges.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for anti_slop_report.json and anti_slop_map.html.",
    )
    parser.add_argument(
        "--cache-path",
        default=".anti-slop-cache/embeddings.json",
        help="Embedding cache path, relative to the scanned root unless absolute.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=300_000,
        help="Skip files larger than this many bytes.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20_000,
        help="Truncate each file to this many decoded characters before embedding.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Maximum analysis items to embed after extraction. Use 0 for no cap.",
    )
    parser.add_argument(
        "--min-symbol-lines",
        type=int,
        default=4,
        help="Skip extracted symbols smaller than this many lines.",
    )
    parser.add_argument(
        "--min-symbol-chars",
        type=int,
        default=160,
        help="Skip extracted symbols smaller than this many characters.",
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=160,
        help="Fallback line-window size for files where symbols cannot be extracted.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=24,
        help="Fallback line-window overlap.",
    )
    parser.add_argument(
        "--extensions",
        default=None,
        help="Comma-separated extension allowlist. Defaults to common text/code files.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories except explicit excludes.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and do not update the embedding cache.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between hosted embedding calls.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=40,
        help="Number of high-similarity pairs to show in CLI output and HTML.",
    )
    parser.add_argument(
        "--max-common-token-ratio",
        type=float,
        default=0.40,
        help=(
            "Ignore tokens appearing in more than this fraction of items when "
            "building near-duplicate candidates. Use 1.0 to index every token."
        ),
    )
    parser.add_argument(
        "--max-near-candidates-per-item",
        type=int,
        default=2000,
        help=(
            "Maximum near-duplicate candidates to score per item after token-index "
            "candidate generation. Use 0 for no cap."
        ),
    )
    parser.add_argument(
        "--slop-example",
        action="append",
        default=[],
        metavar="TEXT_OR_PATH",
        help="Slop example as literal text or a file path. Repeat for multiple examples.",
    )
    parser.add_argument(
        "--slop-examples-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory of slop example files to match against the scanned codebase.",
    )
    parser.add_argument(
        "--slop-match-threshold",
        type=float,
        default=0.74,
        help="Cosine similarity cutoff for example-to-code slop matches.",
    )
    parser.add_argument(
        "--slop-top-matches",
        type=int,
        default=50,
        help="Maximum slop example matches to write and print. Use 0 for all matches.",
    )
    parser.add_argument(
        "--llm-labels",
        action="store_true",
        help="Use an LLM to label clusters and propose review actions. Falls back to heuristics.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("google", "openai"),
        default="google",
        help="Provider for optional cluster labels.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model for optional cluster labels. Defaults to gemini-2.5-flash or gpt-5-mini.",
    )
    return parser.parse_args(argv)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_binary(data: bytes) -> bool:
    if b"\0" in data:
        return True
    if not data:
        return False
    sample = data[:4096]
    textish = sum(1 for b in sample if b in b"\n\r\t\f\b" or 32 <= b <= 126)
    return (textish / len(sample)) < 0.70


def normalize_extension_filter(value: str | None) -> set[str]:
    if not value:
        return set(DEFAULT_EXTENSIONS)
    exts = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        exts.add(item if item.startswith(".") else f".{item}")
    return exts


def should_skip_path(path: Path, root: Path, include_hidden: bool) -> bool:
    rel_parts = path.relative_to(root).parts
    for part in rel_parts[:-1]:
        if part in DEFAULT_EXCLUDE_DIRS:
            return True
        if not include_hidden and part.startswith("."):
            return True
    name = rel_parts[-1]
    if name in DEFAULT_EXCLUDE_FILES:
        return True
    if not include_hidden and name.startswith("."):
        return True
    return False


def summarize_structure(text: str, extension: str) -> tuple[list[str], list[str]]:
    if extension == ".py":
        return summarize_python_structure(text)
    return summarize_text_structure(text)


def summarize_python_structure(text: str) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return summarize_text_structure(text)

    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names if alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            name = call_name(node.func)
            if name:
                calls.add(name)
    return sorted(imports)[:80], sorted(calls)[:120]


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def summarize_text_structure(text: str) -> tuple[list[str], list[str]]:
    imports = set()
    calls = set()
    for match in re.finditer(
        r"""(?mx)
        ^\s*(?:import|from|require\(|use\s+|using\s+|\#include)\s+
        ["'<]?
        ([A-Za-z0-9_./:@-]+)
        """,
        text,
    ):
        imports.add(match.group(1).split("/")[0].split(".")[0])
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_$.]{2,})\s*\(", text):
        name = match.group(1)
        if name not in {"if", "for", "while", "switch", "catch", "return"}:
            calls.add(name)
    return sorted(imports)[:80], sorted(calls)[:120]


def scan_files(args: argparse.Namespace) -> list[FileDoc]:
    root = Path(args.root).expanduser().resolve()
    extensions = normalize_extension_filter(args.extensions)
    docs: list[FileDoc] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if should_skip_path(path, root, args.include_hidden):
            continue
        if path.suffix.lower() not in extensions:
            continue

        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > args.max_file_bytes:
            continue

        try:
            data = path.read_bytes()
        except OSError:
            continue
        if is_binary(data):
            continue

        rel = path.relative_to(root).as_posix()
        text = data.decode("utf-8", errors="replace")
        imports, calls = summarize_structure(text, path.suffix.lower())
        if len(text) > args.max_chars:
            text = text[: args.max_chars]
        docs.append(
            FileDoc(
                path=rel,
                size=size,
                sha256=sha256_bytes(data),
                extension=path.suffix.lower(),
                text=text,
                imports=imports,
                calls=calls,
            )
        )
    return docs


def load_slop_examples(args: argparse.Namespace, root: Path) -> list[SlopExample]:
    examples: list[SlopExample] = []
    for value in args.slop_example:
        path = resolve_existing_input_path(value, root)
        if path and path.is_file():
            example = read_slop_example_file(path, len(examples) + 1, args)
            if example:
                examples.append(example)
        elif path and path.is_dir():
            examples.extend(read_slop_example_dir(path, args, len(examples) + 1))
        else:
            text = str(value).strip()
            if text:
                examples.append(make_literal_slop_example(text, len(examples) + 1))

    for value in args.slop_examples_dir:
        path = Path(value).expanduser()
        if not path.is_absolute():
            cwd_candidate = Path.cwd() / path
            root_candidate = root / path
            path = cwd_candidate if cwd_candidate.exists() else root_candidate
        if not path.exists() or not path.is_dir():
            print(f"Skipping missing slop examples dir: {value}", file=sys.stderr)
            continue
        examples.extend(read_slop_example_dir(path.resolve(), args, len(examples) + 1))

    seen_hashes: set[str] = set()
    unique: list[SlopExample] = []
    for example in examples:
        if example.sha256 in seen_hashes:
            continue
        seen_hashes.add(example.sha256)
        unique.append(example)
    return unique


def resolve_existing_input_path(value: str, root: Path) -> Path | None:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def read_slop_example_dir(path: Path, args: argparse.Namespace, start_index: int) -> list[SlopExample]:
    extensions = normalize_extension_filter(args.extensions)
    examples: list[SlopExample] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if should_skip_path(candidate, path, args.include_hidden):
            continue
        if candidate.suffix.lower() not in extensions:
            continue
        example = read_slop_example_file(candidate, start_index + len(examples), args)
        if example:
            examples.append(example)
    return examples


def read_slop_example_file(
    path: Path, index: int, args: argparse.Namespace
) -> SlopExample | None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(f"Skipping unreadable slop example {path}: {exc}", file=sys.stderr)
        return None
    if len(data) > args.max_file_bytes:
        print(f"Skipping oversized slop example {path}", file=sys.stderr)
        return None
    if is_binary(data):
        print(f"Skipping binary slop example {path}", file=sys.stderr)
        return None
    text = data.decode("utf-8", errors="replace")
    if len(text) > args.max_chars:
        text = text[: args.max_chars]
    return SlopExample(
        id=f"slop-example:{index}:{sha256_bytes(data)[:12]}",
        name=path.stem or f"example-{index}",
        source=path.as_posix(),
        size=len(data),
        sha256=sha256_bytes(data),
        text=text,
    )


def make_literal_slop_example(text: str, index: int) -> SlopExample:
    encoded = text.encode("utf-8")
    words = split_name(text[:80])
    name = "-".join(words[:5]) if words else f"example-{index}"
    return SlopExample(
        id=f"slop-example:{index}:{sha256_bytes(encoded)[:12]}",
        name=name,
        source="literal",
        size=len(encoded),
        sha256=sha256_bytes(encoded),
        text=text,
    )


def build_slop_example_items(examples: Sequence[SlopExample]) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    for example in examples:
        extension = Path(example.source).suffix.lower() or ".txt"
        imports, calls = summarize_structure(example.text, extension)
        items.append(
            SemanticItem(
                id=example.id,
                path=example.source,
                kind="slop-example",
                name=example.name,
                start_line=1,
                end_line=line_count(example.text),
                size=example.size,
                sha256=example.sha256,
                normalized_hash=normalized_source_hash(example.text),
                extension=extension,
                text=example.text,
                imports=imports,
                calls=calls,
            )
        )
    return items


def normalized_source_hash(text: str) -> str:
    normalized_lines = []
    in_block_comment = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if in_block_comment:
            if "*/" in line:
                in_block_comment = False
                line = line.split("*/", 1)[1].strip()
            else:
                continue
        if line.startswith("/*"):
            in_block_comment = "*/" not in line
            continue
        if line.startswith(("#", "//", "--", "*")):
            continue
        line = re.sub(r"\s+", " ", line)
        normalized_lines.append(line)
    normalized = "\n".join(normalized_lines)
    return sha256_text(normalized) if normalized else ""


def item_label(item: SemanticItem) -> str:
    if item.kind == "file":
        return item.path
    suffix = f":{item.start_line}-{item.end_line}" if item.start_line else ""
    return f"{item.path}::{item.kind}:{item.name}{suffix}"


def embedding_text(item: SemanticItem) -> str:
    return (
        "File path: {path}\n"
        "Item kind: {kind}\n"
        "Item name: {name}\n"
        "Line range: {start_line}-{end_line}\n"
        "Imports: {imports}\n"
        "Calls: {calls}\n"
        "Purpose: identify implementation responsibility, behavior, duplication, and overlap.\n\n"
        "{text}"
    ).format(
        path=item.path,
        kind=item.kind,
        name=item.name or "(none)",
        start_line=item.start_line,
        end_line=item.end_line,
        imports=", ".join(item.imports[:25]) or "(none)",
        calls=", ".join(item.calls[:35]) or "(none)",
        text=item.text,
    )


def build_semantic_items(docs: Sequence[FileDoc], args: argparse.Namespace) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    for doc in docs:
        if args.granularity in {"file", "both"}:
            items.append(make_item(doc, "file", Path(doc.path).stem, 1, line_count(doc.text), doc.text))
        if args.granularity in {"symbol", "both"}:
            symbols = extract_symbols(doc, args)
            if not symbols:
                symbols = fallback_chunks(doc, args)
            items.extend(symbols)
    return items


def make_item(
    doc: FileDoc, kind: str, name: str, start_line: int, end_line: int, text: str
) -> SemanticItem:
    item_id = doc.path if kind == "file" else f"{doc.path}::{kind}:{name}:{start_line}-{end_line}"
    encoded = text.encode("utf-8", errors="replace")
    return SemanticItem(
        id=item_id,
        path=doc.path,
        kind=kind,
        name=name,
        start_line=start_line,
        end_line=end_line,
        size=len(encoded),
        sha256=sha256_bytes(encoded),
        normalized_hash=normalized_source_hash(text),
        extension=doc.extension,
        text=text,
        imports=doc.imports,
        calls=doc.calls,
    )


def line_count(text: str) -> int:
    return max(1, text.count("\n") + 1)


def extract_symbols(doc: FileDoc, args: argparse.Namespace) -> list[SemanticItem]:
    if doc.extension == ".py":
        return extract_python_symbols(doc, args)
    return extract_regex_symbols(doc, args)


def extract_python_symbols(doc: FileDoc, args: argparse.Namespace) -> list[SemanticItem]:
    try:
        tree = ast.parse(doc.text)
    except SyntaxError:
        return []
    lines = doc.text.splitlines()
    items: list[SemanticItem] = []

    def visit_body(body: Sequence[ast.stmt], parents: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                add_node(node, "class", parents)
                visit_body(node.body, parents + (node.name,))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parents else "function"
                add_node(node, kind, parents)
                visit_body(node.body, parents + (node.name,))

    def add_node(node: ast.AST, kind: str, parents: tuple[str, ...]) -> None:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        if end - start + 1 < args.min_symbol_lines:
            return
        snippet = "\n".join(lines[start - 1 : end])
        if len(snippet.strip()) < args.min_symbol_chars:
            return
        name = ".".join(parents + (getattr(node, "name", "symbol"),))
        items.append(make_item(doc, kind, name, start, end, snippet))

    visit_body(tree.body, ())
    return items


def extract_regex_symbols(doc: FileDoc, args: argparse.Namespace) -> list[SemanticItem]:
    lines = doc.text.splitlines()
    candidates: list[tuple[int, str, str]] = []
    patterns = [
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", "function"),
        (r"^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)\b", "class"),
        (r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\(?", "function"),
        (r"^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*(?:async\s*)?\(?[^=]*=>", "function"),
        (r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", "function"),
        (r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:async\s+)?[A-Za-z0-9_<>,\[\]?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", "function"),
    ]
    for index, line in enumerate(lines, start=1):
        for pattern, kind in patterns:
            match = re.match(pattern, line)
            if match:
                candidates.append((index, kind, match.group(1)))
                break

    items: list[SemanticItem] = []
    for idx, (start, kind, name) in enumerate(candidates):
        next_start = candidates[idx + 1][0] if idx + 1 < len(candidates) else len(lines) + 1
        end = min(next_start - 1, start + args.chunk_lines - 1)
        snippet = "\n".join(lines[start - 1 : end])
        if end - start + 1 < args.min_symbol_lines or len(snippet.strip()) < args.min_symbol_chars:
            continue
        items.append(make_item(doc, kind, name, start, end, snippet))
    return items


def fallback_chunks(doc: FileDoc, args: argparse.Namespace) -> list[SemanticItem]:
    lines = doc.text.splitlines()
    if not lines:
        return []
    if len(lines) <= args.chunk_lines:
        return [make_item(doc, "chunk", Path(doc.path).stem, 1, len(lines), doc.text)]
    step = max(1, args.chunk_lines - args.chunk_overlap)
    items: list[SemanticItem] = []
    chunk_id = 1
    for start_idx in range(0, len(lines), step):
        end_idx = min(len(lines), start_idx + args.chunk_lines)
        snippet = "\n".join(lines[start_idx:end_idx])
        if len(snippet.strip()) >= args.min_symbol_chars:
            items.append(
                make_item(
                    doc,
                    "chunk",
                    f"{Path(doc.path).stem}-{chunk_id}",
                    start_idx + 1,
                    end_idx,
                    snippet,
                )
            )
            chunk_id += 1
        if end_idx == len(lines):
            break
    return items


class EmbeddingProvider:
    provider_name = "base"

    def cache_identity(self) -> str:
        raise NotImplementedError

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class HashNgramProvider(EmbeddingProvider):
    provider_name = "hash"

    def __init__(self, dims: int) -> None:
        self.dims = dims

    def cache_identity(self) -> str:
        return f"hash-ngram:{self.dims}"

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text).tolist() for text in texts]

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dims, dtype=np.float64)
        tokens = re.findall(r"[A-Za-z0-9_./:-]+", text.lower())
        for token in tokens:
            split_parts = re.findall(r"[a-z]+|[0-9]+", token.replace("_", " "))
            grams = [token]
            grams.extend(split_parts)
            grams.extend(token[i : i + 4] for i in range(max(0, len(token) - 3)))
            for gram in grams:
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dims
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[bucket] += sign
        return normalize_vector(vec)


class GoogleProvider(EmbeddingProvider):
    provider_name = "google"

    def __init__(self, model: str, dims: int, sleep: float) -> None:
        self.model = model
        self.dims = dims
        self.sleep = sleep

    def cache_identity(self) -> str:
        return f"google:{self.model}:{self.dims}:clustering"

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: install google-genai or run `pip install -e .`."
            ) from exc

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY, or use `--provider hash` for a smoke test."
            )

        client = genai.Client(api_key=api_key)
        vectors: list[list[float]] = []
        for index, text in enumerate(texts, start=1):
            content = self._format_text(text)
            config_kwargs: dict[str, object] = {"output_dimensionality": self.dims}
            if self.model == "gemini-embedding-001":
                config_kwargs["task_type"] = "CLUSTERING"
            result = client.models.embed_content(
                model=self.model,
                contents=content,
                config=types.EmbedContentConfig(**config_kwargs),
            )
            if not result.embeddings:
                raise RuntimeError(f"No embedding returned for item {index}.")
            vectors.append(list(result.embeddings[0].values))
            if self.sleep and index < len(texts):
                time.sleep(self.sleep)
        return vectors

    def _format_text(self, text: str) -> str:
        if self.model == "gemini-embedding-2":
            return f"task: clustering | query: {text}"
        return text


class OpenAIProvider(EmbeddingProvider):
    provider_name = "openai"

    def __init__(self, model: str, dims: int) -> None:
        self.model = model
        self.dims = dims

    def cache_identity(self) -> str:
        return f"openai:{self.model}:{self.dims}:clustering"

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Missing dependency: install openai or run `pip install -e .`.") from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY, or use `--provider hash` for a smoke test.")
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dims,
            encoding_format="float",
        )
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]


class OpenRouterProvider(EmbeddingProvider):
    provider_name = "openrouter"

    def __init__(self, model: str, dims: int) -> None:
        self.model = model
        self.dims = dims

    def cache_identity(self) -> str:
        return f"openrouter:{self.model}:{self.dims}:clustering"

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Missing dependency: install openai or run `pip install -e .`.") from exc
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENROUTER_API_KEY, or use `--provider hash` for a smoke test.")
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dims,
            encoding_format="float",
        )
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]


class SentenceTransformerProvider(EmbeddingProvider):
    provider_name = "sentence-transformers"

    def __init__(self, model: str, dims: int) -> None:
        self.model = model
        self.dims = dims

    def cache_identity(self) -> str:
        return f"sentence-transformers:{self.model}:{self.dims}:clustering"

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: run `pip install -e \".[local]\"` for local models."
            ) from exc

        model = SentenceTransformer(self.model)
        prepared = [self._format_text(text) for text in texts]
        kwargs = {"normalize_embeddings": True, "show_progress_bar": True}
        try:
            embeddings = model.encode(prepared, truncate_dim=self.dims, **kwargs)
        except TypeError:
            embeddings = model.encode(prepared, **kwargs)
            if embeddings.shape[1] > self.dims:
                embeddings = embeddings[:, : self.dims]
        return embeddings.tolist()

    def _format_text(self, text: str) -> str:
        if "Qwen3-Embedding" in self.model:
            return (
                "Instruct: Group codebase files by implementation responsibility and "
                f"detect repeated behavior.\nQuery: {text}"
            )
        return text


def make_provider(args: argparse.Namespace) -> EmbeddingProvider:
    dims = resolve_output_dim(args)
    if args.provider == "google":
        return GoogleProvider(args.model or "gemini-embedding-2", dims, args.sleep)
    if args.provider == "openai":
        return OpenAIProvider(args.model or "text-embedding-3-large", dims)
    if args.provider == "openrouter":
        return OpenRouterProvider(args.model or "openai/text-embedding-3-small", dims)
    if args.provider == "sentence-transformers":
        return SentenceTransformerProvider(args.model or "Qwen/Qwen3-Embedding-8B", dims)
    return HashNgramProvider(dims)


def resolve_output_dim(args: argparse.Namespace) -> int:
    if args.output_dim:
        return args.output_dim
    if args.provider == "sentence-transformers":
        model = args.model or "Qwen/Qwen3-Embedding-8B"
        if "Qwen3-Embedding-8B" in model:
            return 4096
        if "Qwen3-Embedding-4B" in model:
            return 2560
        if "Qwen3-Embedding-0.6B" in model:
            return 1024
    if args.provider == "openrouter":
        model = args.model or "openai/text-embedding-3-small"
        if model.endswith("text-embedding-3-small"):
            return 1536
    return 3072


def resolve_cache_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def load_cache(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "entries": {}}


def save_cache(path: Path, cache: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def get_embeddings(
    docs: Sequence[SemanticItem], provider: EmbeddingProvider, args: argparse.Namespace
) -> np.ndarray:
    root = Path(args.root).expanduser().resolve()
    cache_path = resolve_cache_path(root, args.cache_path)
    cache = {"version": 1, "entries": {}} if args.no_cache else load_cache(cache_path)
    entries: dict[str, object] = cache.setdefault("entries", {})  # type: ignore[assignment]

    texts = [embedding_text(doc) for doc in docs]
    cache_keys = [
        sha256_text(
            json.dumps(
                {
                    "provider": provider.cache_identity(),
                    "item_sha256": doc.sha256,
                    "item_id": doc.id,
                    "text_sha256": sha256_text(text),
                },
                sort_keys=True,
            )
        )
        for doc, text in zip(docs, texts)
    ]

    vectors: list[list[float] | None] = []
    misses: list[int] = []
    for key in cache_keys:
        cached = entries.get(key)
        if isinstance(cached, list):
            vectors.append([float(value) for value in cached])
        else:
            vectors.append(None)
            misses.append(len(vectors) - 1)

    if misses:
        missing_texts = [texts[i] for i in misses]
        print(f"Embedding {len(misses)} uncached file(s) with {provider.cache_identity()}...")
        embedded = provider.embed_many(missing_texts)
        for idx, vector in zip(misses, embedded):
            normalized = normalize_vector(np.array(vector, dtype=np.float64)).tolist()
            vectors[idx] = normalized
            entries[cache_keys[idx]] = normalized
        if not args.no_cache:
            save_cache(cache_path, cache)
    else:
        print(f"Loaded {len(vectors)} embedding(s) from cache.")

    matrix = np.array(vectors, dtype=np.float64)
    return normalize_matrix(matrix)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        return vec
    return vec / norm


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_pairs(
    docs: Sequence[SemanticItem],
    embeddings: np.ndarray,
    threshold: float,
    top_limit: int,
) -> tuple[list[SimilarPair], dict[str, int]]:
    similarity = embeddings @ embeddings.T
    pairs_by_key: dict[tuple[str, str, str], SimilarPair] = {}
    top_candidates: list[tuple[float, int, int]] = []
    scored_pairs = 0
    redundant_pairs = 0
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if redundant_parent_pair(docs[i], docs[j]):
                redundant_pairs += 1
                continue
            scored_pairs += 1
            score = float(similarity[i, j])
            if score >= threshold:
                pair = SimilarPair(
                    a=docs[i].id,
                    b=docs[j].id,
                    similarity=score,
                    relation="semantic",
                    evidence="embedding cosine similarity",
                )
                pairs_by_key[pair_key(pair.a, pair.b, pair.relation)] = pair
            if top_limit > 0:
                top_candidates.append((score, i, j))
                if len(top_candidates) > top_limit * 4:
                    top_candidates.sort(reverse=True)
                    del top_candidates[top_limit:]
    if top_limit > 0:
        top_candidates.sort(reverse=True)
        for score, i, j in top_candidates[:top_limit]:
            pair = SimilarPair(
                a=docs[i].id,
                b=docs[j].id,
                similarity=score,
                relation="semantic",
                evidence="embedding cosine similarity",
            )
            pairs_by_key.setdefault(pair_key(pair.a, pair.b, pair.relation), pair)
    pairs = list(pairs_by_key.values())
    pairs.sort(key=lambda pair: pair.similarity, reverse=True)
    return pairs, {
        "semantic_pairs_scored": scored_pairs,
        "semantic_pairs_redundant": redundant_pairs,
        "semantic_pairs_retained": len(pairs),
    }


def duplicate_pairs(
    docs: Sequence[SemanticItem],
    threshold: float,
    max_common_token_ratio: float,
    max_near_candidates_per_item: int,
) -> tuple[list[SimilarPair], dict[str, int]]:
    pairs: list[SimilarPair] = []
    hash_groups: dict[str, list[SemanticItem]] = {}
    for doc in docs:
        if doc.normalized_hash:
            hash_groups.setdefault(doc.normalized_hash, []).append(doc)
    seen: set[tuple[str, str, str]] = set()
    for group in hash_groups.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if redundant_parent_pair(group[i], group[j]):
                    continue
                key = pair_key(group[i].id, group[j].id, "exact")
                seen.add(key)
                pairs.append(
                    SimilarPair(
                        a=group[i].id,
                        b=group[j].id,
                        similarity=1.0,
                        relation="exact",
                        evidence="same normalized source hash",
                    )
                )

    token_sets = [token_set(doc.text) for doc in docs]
    eligible_indices = [index for index, tokens in enumerate(token_sets) if len(tokens) >= 18]
    token_counts: Counter[str] = Counter()
    for index in eligible_indices:
        token_counts.update(token_sets[index])
    max_common_count = max(2, math.ceil(len(eligible_indices) * max_common_token_ratio))
    common_tokens = {token for token, count in token_counts.items() if count > max_common_count}
    token_index: dict[str, list[int]] = {}
    for index in eligible_indices:
        for token in token_sets[index] - common_tokens:
            token_index.setdefault(token, []).append(index)

    candidate_checks = 0
    candidate_pairs = 0
    capped_items = 0
    for i in range(len(docs)):
        if len(token_sets[i]) < 18:
            continue
        candidate_counts: Counter[int] = Counter()
        for token in token_sets[i] - common_tokens:
            for j in token_index.get(token, []):
                if j > i:
                    candidate_counts[j] += 1
        if max_near_candidates_per_item > 0 and len(candidate_counts) > max_near_candidates_per_item:
            candidate_counts = Counter(dict(candidate_counts.most_common(max_near_candidates_per_item)))
            capped_items += 1
        candidate_pairs += len(candidate_counts)
        for j in candidate_counts:
            if redundant_parent_pair(docs[i], docs[j]):
                continue
            if len(token_sets[j]) < 18:
                continue
            key = pair_key(docs[i].id, docs[j].id, "near")
            if key in seen:
                continue
            candidate_checks += 1
            score = jaccard(token_sets[i], token_sets[j])
            if score >= threshold:
                pairs.append(
                    SimilarPair(
                        a=docs[i].id,
                        b=docs[j].id,
                        similarity=score,
                        relation="near",
                        evidence="high token-set overlap",
                    )
                )
    pairs.sort(key=lambda pair: (pair.relation == "exact", pair.similarity), reverse=True)
    return pairs, {
        "near_duplicate_candidate_pairs": candidate_pairs,
        "near_duplicate_candidate_checks": candidate_checks,
        "near_duplicate_common_tokens_skipped": len(common_tokens),
        "near_duplicate_capped_items": capped_items,
    }


def match_slop_examples(
    examples: Sequence[SlopExample],
    example_items: Sequence[SemanticItem],
    docs: Sequence[SemanticItem],
    doc_embeddings: np.ndarray,
    example_embeddings: np.ndarray,
    args: argparse.Namespace,
) -> list[SlopMatch]:
    if not examples or not example_items:
        return []

    semantic_scores = example_embeddings @ doc_embeddings.T
    doc_token_sets = [token_set(doc.text) for doc in docs]
    example_token_sets = [token_set(example.text) for example in example_items]
    example_by_id = {example.id: example for example in examples}
    matches: list[SlopMatch] = []

    for example_index, example in enumerate(example_items):
        for doc_index, doc in enumerate(docs):
            semantic_score = float(semantic_scores[example_index, doc_index])
            relation = ""
            score = semantic_score
            evidence = ""

            if example.normalized_hash and example.normalized_hash == doc.normalized_hash:
                relation = "exact-example"
                score = 1.0
                evidence = "same normalized source hash as slop example"
            else:
                overlap = jaccard(example_token_sets[example_index], doc_token_sets[doc_index])
                if overlap >= args.near_duplicate_threshold:
                    relation = "near-example"
                    score = overlap
                    evidence = "high token-set overlap with slop example"
                elif semantic_score >= args.slop_match_threshold:
                    relation = "semantic-example"
                    evidence = "embedding similarity to slop example"

            if relation:
                source = example_by_id.get(example.id)
                matches.append(
                    SlopMatch(
                        example_id=example.id,
                        example_name=source.name if source else example.name,
                        item_id=doc.id,
                        path=doc.path,
                        kind=doc.kind,
                        name=doc.name,
                        start_line=doc.start_line,
                        end_line=doc.end_line,
                        similarity=score,
                        relation=relation,
                        evidence=evidence,
                    )
                )

    matches.sort(key=slop_match_sort_key, reverse=True)
    if args.slop_top_matches > 0:
        return matches[: args.slop_top_matches]
    return matches


def slop_match_sort_key(match: SlopMatch) -> tuple[int, float]:
    priority = {"exact-example": 3, "near-example": 2, "semantic-example": 1}.get(
        match.relation, 0
    )
    return priority, match.similarity


def redundant_parent_pair(a: SemanticItem, b: SemanticItem) -> bool:
    if a.path != b.path:
        return False
    if a.kind == "file" or b.kind == "file":
        return True
    a_contains_b = a.start_line <= b.start_line and a.end_line >= b.end_line
    b_contains_a = b.start_line <= a.start_line and b.end_line >= a.end_line
    return a_contains_b or b_contains_a


def token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[0-9]{2,}", text.lower())
        if token not in {"the", "and", "for", "with", "from", "return", "const", "class", "function"}
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_key(a: str, b: str, relation: str) -> tuple[str, str, str]:
    first, second = sorted((a, b))
    return first, second, relation


def merge_pairs(semantic_pairs: Sequence[SimilarPair], duplicate_edges: Sequence[SimilarPair]) -> list[SimilarPair]:
    merged: dict[tuple[str, str, str], SimilarPair] = {}
    for pair in list(semantic_pairs) + list(duplicate_edges):
        key = pair_key(pair.a, pair.b, pair.relation)
        existing = merged.get(key)
        if existing is None or pair.similarity > existing.similarity:
            merged[key] = pair
    pairs = list(merged.values())
    pairs.sort(key=lambda pair: pair_sort_key(pair), reverse=True)
    return pairs


def pair_sort_key(pair: SimilarPair) -> tuple[int, float]:
    priority = {"exact": 3, "near": 2, "semantic": 1}.get(pair.relation, 0)
    return priority, pair.similarity


def build_clusters(
    docs: Sequence[SemanticItem],
    semantic_pairs: Sequence[SimilarPair],
    duplicate_edges: Sequence[SimilarPair],
    threshold: float,
) -> list[Cluster]:
    n = len(docs)
    index_by_id = {doc.id: index for index, doc in enumerate(docs)}
    adjacency = [[] for _ in range(n)]
    edge_by_pair: dict[tuple[str, str], list[SimilarPair]] = {}

    def add_edge(pair: SimilarPair) -> None:
        if pair.a == pair.b:
            return
        if pair.a not in index_by_id or pair.b not in index_by_id:
            return
        i, j = index_by_id[pair.a], index_by_id[pair.b]
        adjacency[i].append(j)
        adjacency[j].append(i)
        key = tuple(sorted((pair.a, pair.b)))
        edge_by_pair.setdefault(key, []).append(pair)

    for pair in semantic_pairs:
        if pair.similarity >= threshold:
            add_edge(pair)
    for pair in duplicate_edges:
        add_edge(pair)

    seen = set()
    clusters: list[Cluster] = []
    for start in range(n):
        if start in seen or not adjacency[start]:
            continue
        stack = [start]
        component: list[int] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(component) < 2:
            continue
        component_ids = {docs[index].id for index in component}
        edges = [
            edge
            for ids, pair_edges in edge_by_pair.items()
            if ids[0] in component_ids and ids[1] in component_ids
            for edge in pair_edges
        ]
        sims = [edge.similarity for edge in edges]
        possible_edges = len(component) * (len(component) - 1) / 2
        component_items = [docs[i] for i in sorted(component, key=lambda idx: item_label(docs[idx]))]
        relation_counts = Counter(edge.relation for edge in edges)
        shared_imports = shared_terms([item.imports for item in component_items], 8)
        shared_calls = shared_terms([item.calls for item in component_items], 10)
        label = heuristic_cluster_label(component_items, shared_imports, shared_calls)
        clusters.append(
            Cluster(
                id=len(clusters) + 1,
                label=label,
                items=[item.id for item in component_items],
                files=sorted({item.path for item in component_items}),
                density=len({tuple(sorted((edge.a, edge.b))) for edge in edges}) / possible_edges
                if possible_edges
                else 0.0,
                max_similarity=max(sims) if sims else 0.0,
                semantic_edges=relation_counts.get("semantic", 0),
                exact_edges=relation_counts.get("exact", 0),
                near_edges=relation_counts.get("near", 0),
                shared_imports=shared_imports,
                shared_calls=shared_calls,
                recommendation=review_recommendation(relation_counts, component_items),
            )
        )

    clusters.sort(key=lambda cluster: (len(cluster.files), cluster.max_similarity), reverse=True)
    for index, cluster in enumerate(clusters, start=1):
        cluster.id = index
    return clusters


def shared_terms(term_lists: Sequence[Sequence[str]], limit: int) -> list[str]:
    if len(term_lists) < 2:
        return []
    counter: Counter[str] = Counter()
    for terms in term_lists:
        counter.update(set(terms))
    min_count = max(2, math.ceil(len(term_lists) * 0.4))
    return [term for term, count in counter.most_common() if count >= min_count][:limit]


def heuristic_cluster_label(
    items: Sequence[SemanticItem], shared_imports: Sequence[str], shared_calls: Sequence[str]
) -> str:
    names = []
    for item in items:
        names.extend(split_name(item.name))
        names.extend(split_name(Path(item.path).stem))
    noise = {"test", "spec", "index", "main", "utils", "helper", "helpers", "file", "chunk"}
    common = [word for word, _ in Counter(word for word in names if word not in noise).most_common(4)]
    if common:
        return " ".join(common).title()
    if shared_imports:
        return f"{shared_imports[0]} related code"
    if shared_calls:
        return f"{shared_calls[0]} call pattern"
    parent = common_parent([item.path for item in items])
    return parent or "Related implementation"


def split_name(value: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).replace("_", " ").replace("-", " ")
    return [part.lower() for part in re.findall(r"[A-Za-z0-9]{3,}", parts)]


def common_parent(paths: Sequence[str]) -> str:
    if not paths:
        return ""
    split_paths = [Path(path).parts[:-1] for path in paths]
    shared = []
    for parts in zip(*split_paths):
        if len(set(parts)) == 1:
            shared.append(parts[0])
        else:
            break
    return "/".join(shared)


def review_recommendation(counts: Counter[str], items: Sequence[SemanticItem]) -> str:
    if counts.get("exact", 0):
        return "Review exact normalized duplicates first; one implementation may be removable after checking callers and tests."
    if counts.get("near", 0):
        return "Review near-duplicate items for extracted shared helpers or a single canonical implementation."
    if len({item.path for item in items}) == 1:
        return "Related symbols live in the same file; inspect for local factoring before moving code."
    return "Review the shared imports/calls and responsibilities before consolidating; semantic similarity alone is advisory."


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 0:
        return np.zeros((0, 2))
    if len(embeddings) == 1:
        return np.zeros((1, 2))
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    except np.linalg.LinAlgError:
        coords = np.zeros((len(embeddings), 2))
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    return coords[:, :2]


def cluster_lookup(clusters: Sequence[Cluster]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for cluster in clusters:
        for item_id in cluster.items:
            lookup[item_id] = cluster.id
    return lookup


def write_report(
    output_dir: Path,
    root: Path,
    files: Sequence[FileDoc],
    docs: Sequence[SemanticItem],
    slop_examples: Sequence[SlopExample],
    slop_matches: Sequence[SlopMatch],
    provider: EmbeddingProvider,
    args: argparse.Namespace,
    pairs: Sequence[SimilarPair],
    clusters: Sequence[Cluster],
    coords: np.ndarray,
    analysis_stats: dict[str, int],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "anti_slop_report.json"
    clustered = cluster_lookup(clusters)
    report = {
        "root": str(root),
        "provider": provider.cache_identity(),
        "granularity": args.granularity,
        "threshold": args.threshold,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "slop_match_threshold": args.slop_match_threshold,
        "file_count": len(files),
        "item_count": len(docs),
        "cluster_count": len(clusters),
        "slop_example_count": len(slop_examples),
        "slop_match_count": len(slop_matches),
        "analysis_stats": analysis_stats,
        "scan_limits": {
            "max_items": args.max_items,
            "max_common_token_ratio": args.max_common_token_ratio,
            "max_near_candidates_per_item": args.max_near_candidates_per_item,
        },
        "files": [
            {
                "path": doc.path,
                "size": doc.size,
                "sha256": doc.sha256,
                "extension": doc.extension,
            }
            for doc in files
        ],
        "items": [
            {
                "id": doc.id,
                "path": doc.path,
                "kind": doc.kind,
                "name": doc.name,
                "start_line": doc.start_line,
                "end_line": doc.end_line,
                "size": doc.size,
                "sha256": doc.sha256,
                "normalized_hash": doc.normalized_hash,
                "extension": doc.extension,
                "imports": doc.imports,
                "calls": doc.calls,
                "cluster_id": clustered.get(doc.id),
                "x": float(coords[index, 0]) if len(coords) else 0.0,
                "y": float(coords[index, 1]) if len(coords) else 0.0,
            }
            for index, doc in enumerate(docs)
        ],
        "slop_examples": [
            {
                "id": example.id,
                "name": example.name,
                "source": example.source,
                "size": example.size,
                "sha256": example.sha256,
            }
            for example in slop_examples
        ],
        "slop_matches": [asdict(match) for match in slop_matches],
        "cluster_edges": [
            asdict(pair)
            for pair in pairs
            if pair.relation != "semantic" or pair.similarity >= args.threshold
        ],
        "top_pairs": [asdict(pair) for pair in pairs[: args.top_pairs]],
        "clusters": [asdict(cluster) for cluster in clusters],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def scale_coords(coords: np.ndarray, width: int, height: int) -> list[tuple[float, float]]:
    if len(coords) == 0:
        return []
    xs = coords[:, 0]
    ys = coords[:, 1]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    pad = 70

    def scale(value: float, lo: float, hi: float, size: int) -> float:
        if math.isclose(lo, hi):
            return size / 2
        return pad + ((value - lo) / (hi - lo)) * (size - 2 * pad)

    return [(scale(float(x), min_x, max_x, width), scale(float(y), min_y, max_y, height)) for x, y in coords]


def write_html(
    output_dir: Path,
    root: Path,
    files: Sequence[FileDoc],
    docs: Sequence[SemanticItem],
    slop_examples: Sequence[SlopExample],
    slop_matches: Sequence[SlopMatch],
    provider: EmbeddingProvider,
    args: argparse.Namespace,
    pairs: Sequence[SimilarPair],
    clusters: Sequence[Cluster],
    coords: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "anti_slop_map.html"
    width, height = 1120, 760
    points = scale_coords(coords, width, height)
    clustered = cluster_lookup(clusters)
    point_by_id = {doc.id: points[index] for index, doc in enumerate(docs)}

    edge_lines = []
    for pair in [
        pair for pair in pairs if pair.relation != "semantic" or pair.similarity >= args.threshold
    ][:180]:
        x1, y1 = point_by_id[pair.a]
        x2, y2 = point_by_id[pair.b]
        width_px = 1 + max(0, pair.similarity - args.threshold) * 10
        color = "#ef4444" if pair.relation == "exact" else "#f97316" if pair.relation == "near" else "#94a3b8"
        edge_lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width_px:.2f}" stroke-opacity="0.50" />'
        )

    circles = []
    labels = []
    for index, doc in enumerate(docs):
        x, y = points[index]
        cid = clustered.get(doc.path, 0)
        cid = clustered.get(doc.id, 0)
        color = COLORS[(cid - 1) % len(COLORS)] if cid else "#64748b"
        radius = 8 if doc.kind == "file" else 6 if cid else 4
        escaped_path = html.escape(item_label(doc))
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            f'stroke="#0f172a" stroke-width="1"><title>{escaped_path}</title></circle>'
        )
        if cid and len(labels) < 140:
            labels.append(
                f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" font-size="11" '
                f'fill="#0f172a">{escaped_path}</text>'
            )

    cluster_sections = []
    for cluster in clusters:
        item_rows = "\n".join(f"<li>{html.escape(item_id)}</li>" for item_id in cluster.items)
        shared = []
        if cluster.shared_imports:
            shared.append(f"<p>Shared imports: <code>{html.escape(', '.join(cluster.shared_imports))}</code></p>")
        if cluster.shared_calls:
            shared.append(f"<p>Shared calls: <code>{html.escape(', '.join(cluster.shared_calls))}</code></p>")
        cluster_sections.append(
            "<section>"
            f"<h3>Cluster {cluster.id}: {html.escape(cluster.label)}</h3>"
            f"<p>{len(cluster.items)} items across {len(cluster.files)} files. "
            f"Edges: {cluster.semantic_edges} semantic, {cluster.near_edges} near, "
            f"{cluster.exact_edges} exact. Max score {cluster.max_similarity:.3f}, "
            f"density {cluster.density:.2f}.</p>"
            f"<p>{html.escape(cluster.recommendation)}</p>"
            f"{''.join(shared)}"
            f"<ul>{item_rows}</ul>"
            "</section>"
        )

    top_pair_rows = []
    for pair in pairs[: args.top_pairs]:
        marker = "edge" if pair.similarity >= args.threshold else "near"
        top_pair_rows.append(
            "<tr>"
            f"<td>{pair.similarity:.3f}</td>"
            f"<td>{html.escape(pair.relation)} {marker}</td>"
            f"<td>{html.escape(pair.a)}</td>"
            f"<td>{html.escape(pair.b)}</td>"
            "</tr>"
        )

    slop_match_rows = []
    for match in slop_matches:
        location = f"{match.path}:{match.start_line}-{match.end_line}"
        slop_match_rows.append(
            "<tr>"
            f"<td>{match.similarity:.3f}</td>"
            f"<td>{html.escape(match.relation)}</td>"
            f"<td>{html.escape(match.example_name)}</td>"
            f"<td>{html.escape(location)}</td>"
            f"<td>{html.escape(match.item_id)}</td>"
            "</tr>"
        )
    if slop_examples:
        slop_section = (
            "<h2>Slop Example Matches</h2>"
            "<table>"
            "<thead><tr><th>Similarity</th><th>Kind</th><th>Example</th>"
            "<th>Location</th><th>Item</th></tr></thead>"
            f"<tbody>{''.join(slop_match_rows)}</tbody>"
            "</table>"
            if slop_match_rows
            else "<h2>Slop Example Matches</h2><p>No code items matched the supplied slop examples.</p>"
        )
    else:
        slop_section = ""

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anti-Slop Cannon Map</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --paper: #f8fafc;
      --panel: #ffffff;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
    }}
    header, main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
      line-height: 1.5;
      margin: 6px 0;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .stat strong {{
      display: block;
      font-size: 22px;
    }}
    .map {{
      width: 100%;
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    svg {{
      min-width: {width}px;
      display: block;
    }}
    text {{
      paint-order: stroke;
      stroke: white;
      stroke-width: 3px;
      stroke-linejoin: round;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      font-size: 13px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{
      background: #e2e8f0;
      font-size: 12px;
      text-transform: uppercase;
      color: #334155;
    }}
    section {{
      margin: 18px 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 20px;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    ul {{
      margin: 0;
      padding-left: 22px;
    }}
    li {{
      margin: 4px 0;
      overflow-wrap: anywhere;
    }}
    code {{
      background: #e2e8f0;
      border-radius: 4px;
      padding: 2px 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Anti-Slop Cannon Map</h1>
    <p><code>{html.escape(str(root))}</code></p>
    <p>Provider: <code>{html.escape(provider.cache_identity())}</code></p>
  </header>
  <main>
    <div class="stats">
      <div class="stat"><strong>{len(files)}</strong><span>files scanned</span></div>
      <div class="stat"><strong>{len(docs)}</strong><span>items embedded</span></div>
      <div class="stat"><strong>{len(clusters)}</strong><span>clusters above threshold</span></div>
      <div class="stat"><strong>{args.threshold:.2f}</strong><span>similarity threshold</span></div>
      <div class="stat"><strong>{sum(1 for pair in pairs if pair.relation != 'semantic' or pair.similarity >= args.threshold)}</strong><span>overlap edges</span></div>
      <div class="stat"><strong>{len(slop_matches)}</strong><span>slop example matches</span></div>
    </div>
    <h2>Map</h2>
    <div class="map">
      <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Semantic file cluster map">
        <rect width="{width}" height="{height}" fill="#ffffff" />
        {''.join(edge_lines)}
        {''.join(circles)}
        {''.join(labels)}
      </svg>
    </div>
    <h2>Top Pairs</h2>
    <table>
      <thead>
        <tr><th>Similarity</th><th>Kind</th><th>File A</th><th>File B</th></tr>
      </thead>
      <tbody>
        {''.join(top_pair_rows)}
      </tbody>
    </table>
    {slop_section}
    <h2>Clusters</h2>
    {''.join(cluster_sections) if cluster_sections else '<p>No clusters crossed the selected threshold.</p>'}
  </main>
</body>
</html>
"""
    html_path.write_text(content, encoding="utf-8")
    return html_path


def maybe_label_clusters_with_llm(
    clusters: Sequence[Cluster], docs: Sequence[SemanticItem], args: argparse.Namespace
) -> None:
    if not args.llm_labels or not clusters:
        return
    items_by_id = {doc.id: doc for doc in docs}
    for cluster in clusters:
        sample_items = [items_by_id[item_id] for item_id in cluster.items if item_id in items_by_id][:8]
        prompt = cluster_label_prompt(cluster, sample_items)
        try:
            label, recommendation = request_cluster_label(prompt, args)
        except Exception as exc:  # noqa: BLE001 - labeling is advisory and should not break analysis.
            print(f"LLM label fallback for cluster {cluster.id}: {exc}", file=sys.stderr)
            continue
        if label:
            cluster.label = label[:90]
        if recommendation:
            cluster.recommendation = recommendation[:240]


def cluster_label_prompt(cluster: Cluster, items: Sequence[SemanticItem]) -> str:
    item_summaries = []
    for item in items:
        snippet = "\n".join(item.text.strip().splitlines()[:30])
        item_summaries.append(
            {
                "id": item.id,
                "kind": item.kind,
                "imports": item.imports[:12],
                "calls": item.calls[:18],
                "snippet": snippet[:1800],
            }
        )
    return (
        "You label code-overlap clusters. Return only JSON with keys label and recommendation. "
        "The recommendation must be cautious and review-oriented, never telling the user to delete "
        "code without checking callers and tests.\n\n"
        + json.dumps(
            {
                "cluster_id": cluster.id,
                "current_label": cluster.label,
                "edge_counts": {
                    "semantic": cluster.semantic_edges,
                    "near": cluster.near_edges,
                    "exact": cluster.exact_edges,
                },
                "shared_imports": cluster.shared_imports,
                "shared_calls": cluster.shared_calls,
                "items": item_summaries,
            },
            indent=2,
        )
    )


def request_cluster_label(prompt: str, args: argparse.Namespace) -> tuple[str, str]:
    if args.llm_provider == "google":
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("install google-genai for Google LLM labels") from exc
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("set GEMINI_API_KEY or GOOGLE_API_KEY for Google LLM labels")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=args.llm_model or "gemini-2.5-flash",
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install openai for OpenAI LLM labels") from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("set OPENAI_API_KEY for OpenAI LLM labels")
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=args.llm_model or "gpt-5-mini",
            input=prompt,
        )
        text = getattr(response, "output_text", "") or ""
    payload = parse_json_object(text)
    return str(payload.get("label", "")), str(payload.get("recommendation", ""))


def parse_json_object(text: str) -> dict[str, object]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def print_summary(
    files: Sequence[FileDoc],
    docs: Sequence[SemanticItem],
    slop_examples: Sequence[SlopExample],
    slop_matches: Sequence[SlopMatch],
    pairs: Sequence[SimilarPair],
    clusters: Sequence[Cluster],
    args: argparse.Namespace,
    report_path: Path,
    html_path: Path,
) -> None:
    print()
    print(f"Scanned files: {len(files)}")
    print(f"Embedded items: {len(docs)}")
    print(f"Clusters above {args.threshold:.2f}: {len(clusters)}")
    if slop_examples:
        print(f"Slop examples: {len(slop_examples)}")
        print(f"Slop matches above {args.slop_match_threshold:.2f}: {len(slop_matches)}")
    print(f"Report: {report_path}")
    print(f"Map: {html_path}")
    if slop_matches:
        print()
        print("Top slop example matches:")
        for match in slop_matches[:10]:
            location = f"{match.path}:{match.start_line}-{match.end_line}"
            print(
                f"  {match.relation:16} {match.similarity:.3f}  "
                f"{match.example_name}  ->  {location}"
            )
    if clusters:
        print()
        print("Largest clusters:")
        for cluster in clusters[:5]:
            preview = ", ".join(cluster.items[:4])
            if len(cluster.items) > 4:
                preview += ", ..."
            print(
                f"  {cluster.id}. {cluster.label}: {len(cluster.items)} items, "
                f"max={cluster.max_similarity:.3f}: {preview}"
            )
    if pairs:
        print()
        print("Top pairs:")
        for pair in pairs[: min(args.top_pairs, 10)]:
            print(f"  {pair.relation:8} {pair.similarity:.3f}  {pair.a}  <->  {pair.b}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Root is not a directory: {root}", file=sys.stderr)
        return 2
    if not 0 <= args.threshold <= 1:
        print("--threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if not 0 <= args.near_duplicate_threshold <= 1:
        print("--near-duplicate-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if not 0 <= args.max_common_token_ratio <= 1:
        print("--max-common-token-ratio must be between 0 and 1", file=sys.stderr)
        return 2
    if args.max_items < 0:
        print("--max-items must be zero or greater", file=sys.stderr)
        return 2
    if args.max_near_candidates_per_item < 0:
        print("--max-near-candidates-per-item must be zero or greater", file=sys.stderr)
        return 2
    if not 0 <= args.slop_match_threshold <= 1:
        print("--slop-match-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.slop_top_matches < 0:
        print("--slop-top-matches must be zero or greater", file=sys.stderr)
        return 2

    provider = make_provider(args)
    files = scan_files(args)
    if not files:
        print("No eligible text/code files found.", file=sys.stderr)
        return 1
    docs = build_semantic_items(files, args)
    if not docs:
        print("No eligible files, symbols, or chunks found.", file=sys.stderr)
        return 1
    extracted_item_count = len(docs)
    if args.max_items and len(docs) > args.max_items:
        docs = docs[: args.max_items]
        print(f"Limiting analysis items: {len(docs)} of {extracted_item_count} extracted.")
    slop_examples = load_slop_examples(args, root)
    slop_items = build_slop_example_items(slop_examples)

    print(f"Scanning root: {root}")
    print(f"Eligible files: {len(files)}")
    print(f"Embedding granularity: {args.granularity} ({len(docs)} items)")
    if slop_examples:
        print(f"Slop examples: {len(slop_examples)}")
    all_embeddings = get_embeddings([*docs, *slop_items], provider, args)
    embeddings = all_embeddings[: len(docs)]
    slop_embeddings = all_embeddings[len(docs) :]
    semantic_pairs, semantic_stats = cosine_pairs(docs, embeddings, args.threshold, args.top_pairs)
    duplicate_edges, duplicate_stats = duplicate_pairs(
        docs,
        args.near_duplicate_threshold,
        args.max_common_token_ratio,
        args.max_near_candidates_per_item,
    )
    analysis_stats = {
        "extracted_item_count": extracted_item_count,
        **semantic_stats,
        **duplicate_stats,
    }
    pairs = merge_pairs(semantic_pairs, duplicate_edges)
    clusters = build_clusters(docs, semantic_pairs, duplicate_edges, args.threshold)
    slop_matches = match_slop_examples(
        slop_examples,
        slop_items,
        docs,
        embeddings,
        slop_embeddings,
        args,
    )
    maybe_label_clusters_with_llm(clusters, docs, args)
    coords = pca_2d(embeddings)
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = write_report(
        output_dir,
        root,
        files,
        docs,
        slop_examples,
        slop_matches,
        provider,
        args,
        pairs,
        clusters,
        coords,
        analysis_stats,
    )
    html_path = write_html(
        output_dir,
        root,
        files,
        docs,
        slop_examples,
        slop_matches,
        provider,
        args,
        pairs,
        clusters,
        coords,
    )
    print_summary(files, docs, slop_examples, slop_matches, pairs, clusters, args, report_path, html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
