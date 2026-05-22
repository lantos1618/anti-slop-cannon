# Anti-Slop Cannon

Visualize a codebase as semantic clusters so repeated or overlapping implementation
responsibilities stand out.

The script scans text/code files, extracts functions/classes/chunks, embeds each item,
adds exact and near-duplicate evidence, builds connected clusters above a threshold,
optionally matches supplied slop examples against the codebase, and writes:

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
anti-slop-cannon /path/to/code \
  --provider google \
  --model gemini-embedding-2 \
  --output-dim 3072
```

For best open-weight local Qwen embeddings:

```bash
pip install -e ".[local]"
anti-slop-cannon /path/to/code \
  --provider sentence-transformers \
  --model Qwen/Qwen3-Embedding-8B \
  --output-dim 4096
```

For OpenAI embeddings:

```bash
export OPENAI_API_KEY=...
anti-slop-cannon /path/to/code \
  --provider openai \
  --model text-embedding-3-large \
  --output-dim 3072
```

For OpenRouter embeddings:

```bash
export OPENROUTER_API_KEY=...
anti-slop-cannon /path/to/code \
  --provider openrouter \
  --model openai/text-embedding-3-small \
  --output-dim 1536
```

For a no-API smoke test:

```bash
anti-slop-cannon . --provider hash --output-dir .anti-slop-out
```

The hash provider is only a cheap lexical fallback. It is useful for checking the
pipeline, but it is not a replacement for semantic embeddings.

## Usage

```bash
anti-slop-cannon /path/to/code \
  --provider google \
  --model gemini-embedding-2 \
  --output-dim 3072 \
  --granularity symbol \
  --threshold 0.82 \
  --near-duplicate-threshold 0.86 \
  --output-dir anti-slop-output
```

Run with examples of slop you already know about:

```bash
anti-slop-cannon /path/to/code \
  --provider google \
  --slop-example examples/copy_paste_cache.py \
  --slop-example "manual JSON string building with regex parsing" \
  --slop-examples-dir examples/slop-patterns \
  --slop-match-threshold 0.74
```

Examples can be literal text, individual files, or directories of files. Matches are
written to `slop_examples` and `slop_matches` in `anti_slop_report.json` and shown in
the HTML report.

## Clustering Method

Anti-Slop Cannon clusters an evidence graph:

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
8. Match supplied slop examples against the embedded code items.
9. Emit review recommendations. The tool never deletes code or treats clusters as proof.

Useful flags:

- `--threshold`: similarity cutoff for duplicate/overlap edges
- `--near-duplicate-threshold`: token-overlap cutoff for near-duplicate evidence
- `--granularity`: `symbol` for functions/classes/chunks, `file` for whole files, `both` for comparison
- `--slop-example`: literal text or a file path to match against the codebase
- `--slop-examples-dir`: directory of example snippets/files to match against the codebase
- `--slop-match-threshold`: similarity cutoff for example-to-code matches
- `--slop-top-matches`: maximum example matches to emit, or `0` for all matches
- `--max-common-token-ratio`: skip overly common tokens when finding near-duplicate candidates
- `--max-near-candidates-per-item`: cap near-duplicate candidate scoring per item for large repos
- `--llm-labels`: optionally ask a Google/OpenAI LLM to label clusters and write review notes
- `--min-symbol-lines`, `--min-symbol-chars`: suppress tiny symbols
- `--chunk-lines`, `--chunk-overlap`: control fallback chunks
- `--max-file-bytes`: skip files above this size
- `--max-chars`: truncate each file before embedding
- `--max-items`: cap extracted items for very large or exploratory scans
- `--include-hidden`: include hidden files/directories
- `--extensions`: comma-separated extension allowlist, for example `.py,.ts,.tsx,.md`
- `--cache-path`: reuse unchanged embeddings across runs

Open the generated `anti_slop_map.html` in a browser. Dense clusters and high-similarity
pairs are the first places to inspect for repeated responsibilities, duplicated
implementations, and files that may want consolidation.

For large repos, start with `--granularity symbol`, keep generated/vendor directories
excluded, and use `--max-items` for an exploratory first pass. Tune
`--max-near-candidates-per-item` downward if the near-duplicate pass is still too broad.
The JSON report includes `analysis_stats` and `scan_limits` so scan cost and candidate
pruning are visible.

## Evals

Run the deterministic fixture eval with no API key:

```bash
python evals/run_evals.py --provider hash
```

For hosted embeddings, put the key in `.env` or export it in the shell. The eval runner
loads `.env` and does not print key values.

```bash
GEMINI_API_KEY=... python evals/run_evals.py --provider google --keep-output
OPENAI_API_KEY=... python evals/run_evals.py --provider openai --keep-output
OPENROUTER_API_KEY=... python evals/run_evals.py --provider openrouter --keep-output
```

The eval fixture expects the cannon to find a duplicated `parse_total_rows`
implementation, produce at least one cluster, and match the supplied slop example
against both duplicated files.

Current fixture result with the offline hash provider:

| Metric | Result |
| --- | ---: |
| Known duplicate pair recall | `1/1` |
| Slop target file recall | `2/2` |
| Files scanned | `4` |
| Items analyzed | `10` |
| Clusters found | `2` |
| Slop matches | `4` |
| Exact edges | `1` |
| Near edges | `1` |
| Semantic edges | `5` |

Smoke scans on five real repositories from `lantos1618` and `lambda-run` produced:

| Repo | Files | Items | Clusters | Clustered Items | Exact | Near | Semantic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `better-ui` | 159 | 1134 | 91 | 64.5% | 67 | 47 | 4657 |
| `lambda-run/deploy.me` | 144 | 905 | 64 | 37.3% | 9 | 127 | 1223 |
| `arxiv.gg` | 68 | 343 | 31 | 30.6% | 3 | 1 | 101 |
| `sumup-rs` | 47 | 272 | 16 | 66.5% | 154 | 3 | 542 |
| `deploy-me-sdk` | 12 | 43 | 2 | 25.6% | 0 | 0 | 20 |
| **Total** | **430** | **2697** | **204** | **50.6%** | **233** | **178** | **6543** |

Representative high-confidence findings:

- `lambda-run/deploy.me`: exact duplicate `eurPerHour` helpers across catalog providers
- `lambda-run/deploy.me`: exact duplicate `useMe` page helpers in dashboard/providers pages
- `better-ui`: exact duplicate demo route helpers across Next.js and Vite examples
- `sumup-rs`: repeated `response` boilerplate across resource modules
- `arxiv.gg`: exact duplicate template blocks and near-duplicate migration helpers

The real-repo smoke scans used `--provider hash`, so exact and near-duplicate counts
are the strongest evidence. Semantic edges are review leads; hosted embeddings should
improve semantic quality once a valid provider key is available.
