# Anti-Slop Canon

Visualize a codebase as semantic clusters so repeated or overlapping files stand out.

The script scans text/code files, embeds each file, computes all-pairs cosine similarity,
builds connected clusters above a threshold, and writes:

- `anti_slop_report.json`: machine-readable files, pairs, clusters, and model settings
- `anti_slop_map.html`: a static browser visualization of the codebase

## Model Choice

Default hosted model: `gemini-embedding-2`.

Google's current Gemini embedding docs list `gemini-embedding-2` as the stable embedding
model updated in April 2026, with text/image/video/audio/PDF input support, 8192 input
tokens, and 128-3072 dimensional outputs. For this text-code clustering use case, the
script formats each file as a clustering task before embedding.

Strong open-weight option: `Qwen/Qwen3-Embedding-8B`.

Qwen's model card reports `Qwen3-Embedding-8B` as an Apache-2.0, 32K-context embedding
model with up to 4096 dimensions, broad multilingual/code retrieval support, and a
70.58 MTEB multilingual score as of its June 2025 leaderboard snapshot. Use it through
the `sentence-transformers` provider when you have enough local GPU/CPU capacity.

Primary references:

- Google Gemini embeddings docs: https://ai.google.dev/gemini-api/docs/embeddings
- Google Gemini Embedding GA post: https://developers.googleblog.com/en/gemini-embedding-available-gemini-api/
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
python anti_slop_canon.py /path/to/code --provider google --model gemini-embedding-2
```

For local Qwen embeddings:

```bash
pip install -e ".[local]"
python anti_slop_canon.py /path/to/code \
  --provider sentence-transformers \
  --model Qwen/Qwen3-Embedding-8B \
  --output-dim 4096
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
  --output-dim 768 \
  --threshold 0.80 \
  --output-dir anti-slop-output
```

Useful flags:

- `--threshold`: similarity cutoff for duplicate/overlap edges
- `--max-file-bytes`: skip files above this size
- `--max-chars`: truncate each file before embedding
- `--include-hidden`: include hidden files/directories
- `--extensions`: comma-separated extension allowlist, for example `.py,.ts,.tsx,.md`
- `--cache-path`: reuse unchanged embeddings across runs

Open the generated `anti_slop_map.html` in a browser. Dense clusters and high-similarity
pairs are the first places to inspect for repeated responsibilities, duplicated
implementations, and files that may want consolidation.

