#!/usr/bin/env python3
"""Semantic clustering map for spotting repeated codebase files."""

from __future__ import annotations

import argparse
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
from typing import Iterable, Sequence

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


@dataclass
class SimilarPair:
    a: str
    b: str
    similarity: float


@dataclass
class Cluster:
    id: int
    files: list[str]
    density: float
    max_similarity: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed each codebase file and visualize semantic overlap clusters."
    )
    parser.add_argument("root", nargs="?", default=".", help="Codebase root to scan.")
    parser.add_argument(
        "--provider",
        choices=("google", "sentence-transformers", "hash"),
        default="google",
        help="Embedding backend. Use hash only for offline smoke tests.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Embedding model. Defaults to gemini-embedding-2 for Google, "
            "Qwen/Qwen3-Embedding-0.6B for sentence-transformers, or hash-ngram."
        ),
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=768,
        help="Embedding dimensions to request/use when the provider supports it.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Cosine similarity cutoff for cluster edges.",
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
        if len(text) > args.max_chars:
            text = text[: args.max_chars]
        docs.append(
            FileDoc(
                path=rel,
                size=size,
                sha256=sha256_bytes(data),
                extension=path.suffix.lower(),
                text=text,
            )
        )
    return docs


def embedding_text(doc: FileDoc) -> str:
    return (
        "File path: {path}\n"
        "File extension: {extension}\n"
        "Purpose: identify implementation responsibility, behavior, and overlap.\n\n"
        "{text}"
    ).format(path=doc.path, extension=doc.extension or "none", text=doc.text)


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

        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise SystemExit(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY, or use `--provider hash` for a smoke test."
            )

        client = genai.Client()
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
    if args.provider == "google":
        return GoogleProvider(args.model or "gemini-embedding-2", args.output_dim, args.sleep)
    if args.provider == "sentence-transformers":
        return SentenceTransformerProvider(
            args.model or "Qwen/Qwen3-Embedding-0.6B", args.output_dim
        )
    return HashNgramProvider(args.output_dim)


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
    docs: Sequence[FileDoc], provider: EmbeddingProvider, args: argparse.Namespace
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
                    "file_sha256": doc.sha256,
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


def cosine_pairs(docs: Sequence[FileDoc], embeddings: np.ndarray) -> tuple[np.ndarray, list[SimilarPair]]:
    similarity = embeddings @ embeddings.T
    pairs: list[SimilarPair] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            pairs.append(
                SimilarPair(
                    a=docs[i].path,
                    b=docs[j].path,
                    similarity=float(similarity[i, j]),
                )
            )
    pairs.sort(key=lambda pair: pair.similarity, reverse=True)
    return similarity, pairs


def build_clusters(
    docs: Sequence[FileDoc], similarity: np.ndarray, threshold: float
) -> list[Cluster]:
    n = len(docs)
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

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
        sims = [
            float(similarity[i, j])
            for offset, i in enumerate(component)
            for j in component[offset + 1 :]
        ]
        above = [value for value in sims if value >= threshold]
        possible_edges = len(component) * (len(component) - 1) / 2
        clusters.append(
            Cluster(
                id=len(clusters) + 1,
                files=[docs[i].path for i in sorted(component, key=lambda idx: docs[idx].path)],
                density=len(above) / possible_edges if possible_edges else 0.0,
                max_similarity=max(sims) if sims else 0.0,
            )
        )

    clusters.sort(key=lambda cluster: (len(cluster.files), cluster.max_similarity), reverse=True)
    for index, cluster in enumerate(clusters, start=1):
        cluster.id = index
    return clusters


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
        for file_path in cluster.files:
            lookup[file_path] = cluster.id
    return lookup


def write_report(
    output_dir: Path,
    root: Path,
    docs: Sequence[FileDoc],
    provider: EmbeddingProvider,
    args: argparse.Namespace,
    pairs: Sequence[SimilarPair],
    clusters: Sequence[Cluster],
    coords: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "anti_slop_report.json"
    clustered = cluster_lookup(clusters)
    report = {
        "root": str(root),
        "provider": provider.cache_identity(),
        "threshold": args.threshold,
        "file_count": len(docs),
        "cluster_count": len(clusters),
        "files": [
            {
                "path": doc.path,
                "size": doc.size,
                "sha256": doc.sha256,
                "extension": doc.extension,
                "cluster_id": clustered.get(doc.path),
                "x": float(coords[index, 0]) if len(coords) else 0.0,
                "y": float(coords[index, 1]) if len(coords) else 0.0,
            }
            for index, doc in enumerate(docs)
        ],
        "similar_pairs": [asdict(pair) for pair in pairs if pair.similarity >= args.threshold],
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
    docs: Sequence[FileDoc],
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
    point_by_path = {doc.path: points[index] for index, doc in enumerate(docs)}

    edge_lines = []
    for pair in [pair for pair in pairs if pair.similarity >= args.threshold][:160]:
        x1, y1 = point_by_path[pair.a]
        x2, y2 = point_by_path[pair.b]
        width_px = 1 + max(0, pair.similarity - args.threshold) * 10
        edge_lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#94a3b8" stroke-width="{width_px:.2f}" stroke-opacity="0.48" />'
        )

    circles = []
    labels = []
    for index, doc in enumerate(docs):
        x, y = points[index]
        cid = clustered.get(doc.path, 0)
        color = COLORS[(cid - 1) % len(COLORS)] if cid else "#64748b"
        radius = 7 if cid else 5
        escaped_path = html.escape(doc.path)
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
        items = "\n".join(f"<li>{html.escape(path)}</li>" for path in cluster.files)
        cluster_sections.append(
            "<section>"
            f"<h3>Cluster {cluster.id}: {len(cluster.files)} files, "
            f"max similarity {cluster.max_similarity:.3f}, density {cluster.density:.2f}</h3>"
            f"<ul>{items}</ul>"
            "</section>"
        )

    top_pair_rows = []
    for pair in pairs[: args.top_pairs]:
        marker = "edge" if pair.similarity >= args.threshold else "near"
        top_pair_rows.append(
            "<tr>"
            f"<td>{pair.similarity:.3f}</td>"
            f"<td>{marker}</td>"
            f"<td>{html.escape(pair.a)}</td>"
            f"<td>{html.escape(pair.b)}</td>"
            "</tr>"
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anti-Slop Canon Map</title>
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
    <h1>Anti-Slop Canon Map</h1>
    <p><code>{html.escape(str(root))}</code></p>
    <p>Provider: <code>{html.escape(provider.cache_identity())}</code></p>
  </header>
  <main>
    <div class="stats">
      <div class="stat"><strong>{len(docs)}</strong><span>files embedded</span></div>
      <div class="stat"><strong>{len(clusters)}</strong><span>clusters above threshold</span></div>
      <div class="stat"><strong>{args.threshold:.2f}</strong><span>similarity threshold</span></div>
      <div class="stat"><strong>{sum(1 for pair in pairs if pair.similarity >= args.threshold)}</strong><span>overlap edges</span></div>
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
    <h2>Clusters</h2>
    {''.join(cluster_sections) if cluster_sections else '<p>No clusters crossed the selected threshold.</p>'}
  </main>
</body>
</html>
"""
    html_path.write_text(content, encoding="utf-8")
    return html_path


def print_summary(
    docs: Sequence[FileDoc],
    pairs: Sequence[SimilarPair],
    clusters: Sequence[Cluster],
    args: argparse.Namespace,
    report_path: Path,
    html_path: Path,
) -> None:
    print()
    print(f"Embedded files: {len(docs)}")
    print(f"Clusters above {args.threshold:.2f}: {len(clusters)}")
    print(f"Report: {report_path}")
    print(f"Map: {html_path}")
    if clusters:
        print()
        print("Largest clusters:")
        for cluster in clusters[:5]:
            preview = ", ".join(cluster.files[:4])
            if len(cluster.files) > 4:
                preview += ", ..."
            print(
                f"  {cluster.id}. {len(cluster.files)} files, "
                f"max={cluster.max_similarity:.3f}: {preview}"
            )
    if pairs:
        print()
        print("Top pairs:")
        for pair in pairs[: min(args.top_pairs, 10)]:
            print(f"  {pair.similarity:.3f}  {pair.a}  <->  {pair.b}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Root is not a directory: {root}", file=sys.stderr)
        return 2
    if not 0 <= args.threshold <= 1:
        print("--threshold must be between 0 and 1", file=sys.stderr)
        return 2

    provider = make_provider(args)
    docs = scan_files(args)
    if not docs:
        print("No eligible text/code files found.", file=sys.stderr)
        return 1

    print(f"Scanning root: {root}")
    print(f"Eligible files: {len(docs)}")
    embeddings = get_embeddings(docs, provider, args)
    similarity, pairs = cosine_pairs(docs, embeddings)
    clusters = build_clusters(docs, similarity, args.threshold)
    coords = pca_2d(embeddings)
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = write_report(output_dir, root, docs, provider, args, pairs, clusters, coords)
    html_path = write_html(output_dir, root, docs, provider, args, pairs, clusters, coords)
    print_summary(docs, pairs, clusters, args, report_path, html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

