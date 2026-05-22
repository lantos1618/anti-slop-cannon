# Anti-Slop Canon

Visualize a codebase as semantic clusters so repeated or overlapping implementation
responsibilities stand out.

The script scans text/code files, extracts functions/classes/chunks, embeds each item,
adds exact and near-duplicate evidence, builds connected clusters above a threshold,
and writes:

- `anti_slop_report.json`: machine-readable files, pairs, clusters, and model settings
- `anti_slop_map.html`: a static browser visualization of the codebase

## Model Choice

Default hosted model: `gemini-embedding-2`.

There is no current OpenAI `text-embedding-4` in the official OpenAI docs, and the
current Gemini docs do not list a Gemini "embedding 4" model. For hosted SOA, use
`gemini-embedding-2` at 3072 dimensions. Google's current Gemini embedding docs list
it as the latest Gemini embedding model, with text/image/video/audio/PDF input support,
8192 input tokens, and 3072-dimensional default output. For this text-code clustering
use case, the script formats each item as a clustering task before embedding.

Best open-weight option: `Qwen/Qwen3-Embedding-8B`.

Qwen's model card reports `Qwen3-Embedding-8B` as an Apache-2.0, 32K-context embedding
model with up to 4096 dimensions, broad multilingual/code retrieval support, and a
70.58 MTEB multilingual score as of its June 2025 leaderboard snapshot. Use it through
the `sentence-transformers` provider when you have enough local GPU/CPU capacity.

OpenAI fallback: `text-embedding-3-large`, which the current OpenAI docs list as
OpenAI's most capable embedding model. It is useful when your stack is already on
OpenAI, but it is not the top benchmark choice for this tool.

Primary references:

- Google Gemini embeddings docs: https://ai.google.dev/gemini-api/docs/embeddings
- Google Gemini Embedding GA post: https://developers.googleblog.com/en/gemini-embedding-available-gemini-api/
- OpenAI embeddings docs: https://platform.openai.com/docs/guides/embeddings
- Qwen3-Embedding-8B model card: https://huggingface.co/Qwen/Qwen3-Embedding-8B
- Qwen3 Embedding paper: https://arxiv.org/abs/2506.05176

## Install

```bash
cd /home/ubuntu/anti-slop
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Google embeddings:

```bash
export GEMINI_API_KEY=...
python anti_slop_canon.py /path/to/code \
  --provider google \
  --model gemini-embedding-2 \
  --output-dim 3072
```

For best open-weight local Qwen embeddings:

```bash
pip install -e ".[local]"
python anti_slop_canon.py /path/to/code \
  --provider sentence-transformers \
  --model Qwen/Qwen3-Embedding-8B \
  --output-dim 4096
```

For OpenAI embeddings:

```bash
export OPENAI_API_KEY=...
python anti_slop_canon.py /path/to/code \
  --provider openai \
  --model text-embedding-3-large \
  --output-dim 3072
```

For a no-API smoke test:

```bash
python anti_slop_canon.py . --provider hash --output-dir .anti-slop-out
```

The hash provider is only a cheap lexical fallback. It is useful for checking the
pipeline, but it is not a replacement for semantic embeddings.

## Usage

```bash
python anti_slop_canon.py /path/to/code \
  --provider google \
  --model gemini-embedding-2 \
  --output-dim 3072 \
  --granularity symbol \
  --threshold 0.82 \
  --near-duplicate-threshold 0.86 \
  --output-dir anti-slop-output
```

## Clustering Method

Anti-Slop Canon clusters an evidence graph:

1. Extract analysis items from each file:
   - Python uses `ast` to extract classes, functions, and methods.
   - JS/TS/Rust/Java-like files use conservative symbol regexes.
   - Files without symbols fall back to line-window chunks.
2. Embed every item with the selected provider.
3. Add a semantic edge when cosine similarity is above `--threshold`.
4. Add an exact edge when normalized source text has the same hash.
5. Add a near edge when token-set overlap is above `--near-duplicate-threshold`.
6. Build connected components over those edges.
7. Label each cluster from symbol names, paths, shared imports, and shared calls.
8. Emit review recommendations. The tool never deletes code or treats clusters as proof.

Useful flags:

- `--threshold`: similarity cutoff for duplicate/overlap edges
- `--near-duplicate-threshold`: token-overlap cutoff for near-duplicate evidence
- `--granularity`: `symbol` for functions/classes/chunks, `file` for whole files, `both` for comparison
- `--llm-labels`: optionally ask a Google/OpenAI LLM to label clusters and write review notes
- `--min-symbol-lines`, `--min-symbol-chars`: suppress tiny symbols
- `--chunk-lines`, `--chunk-overlap`: control fallback chunks
- `--max-file-bytes`: skip files above this size
- `--max-chars`: truncate each file before embedding
- `--include-hidden`: include hidden files/directories
- `--extensions`: comma-separated extension allowlist, for example `.py,.ts,.tsx,.md`
- `--cache-path`: reuse unchanged embeddings across runs

Open the generated `anti_slop_map.html` in a browser. Dense clusters and high-similarity
pairs are the first places to inspect for repeated responsibilities, duplicated
implementations, and files that may want consolidation.
