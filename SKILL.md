---
name: embenx
description: Benchmark and compare vector indexing libraries (FAISS, Chroma, Qdrant, Milvus, LanceDB) using the Embenx CLI. Use this skill whenever the user wants to compare vector databases, benchmark embedding or indexing performance, evaluate search latency or index footprint, choose a vector DB for RAG or semantic search, or run any vector indexing experiment — even if they don't mention "embenx" or "benchmark" explicitly.
---

# Embenx Skill

This skill helps you benchmark and compare vector search backends so the user can make an informed choice for their use case.

## Core Workflow

### 1. Identify Requirements

Ask (or infer from context):
- **Dataset**: HuggingFace dataset name, or local file path (CSV/JSON/JSONL)?
- **Indexers**: Which backends to compare? Default is all: `faiss`, `chroma`, `qdrant`, `milvus`, `lance`.
- **Embedding model**: Local Ollama model (free, no API key) or cloud (OpenAI, Anthropic)?
- **Scale**: How many docs? Start small (100) unless the user specifies otherwise.
- **Key constraint**: Speed, memory, or index size?

### 2. Check Environment

Always run this first — it checks installed indexers and verifies the embedding model is ready:

```bash
uv run embenx setup
```

By default this checks `ollama/nomic-embed-text`. To check a different model:
```bash
uv run embenx setup --model ollama/qwen3-embedding
```

To also pull the Ollama model if it's missing:
```bash
uv run embenx setup --pull
```

If `setup` reports missing indexers, install them individually (e.g., `uv pip install milvus-lite` for Milvus). Fix all issues before proceeding.

### 3. Run the Benchmark

**HuggingFace dataset:**
```bash
uv run embenx benchmark --dataset <name> --text-column <col> --max-docs <num> --model <model>
```

**Local file (CSV/JSON/JSONL):**
```bash
uv run embenx benchmark --dataset json --data-files <path> --text-column <col> --max-docs <num>
```

**Subset of indexers:**
```bash
uv run embenx benchmark --dataset squad --indexers faiss,qdrant --max-docs 200
```

**Compare two embedding models:**
Run twice with different `--model` values (e.g., `ollama/nomic-embed-text` vs `openai/text-embedding-3-small`).

**Key flags:**
| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` / `-d` | required | HF dataset name or `csv`/`json` |
| `--text-column` / `-c` | `text` | Column to embed |
| `--max-docs` / `-m` | `1000` | Use `100` for quick tests |
| `--split` / `-s` | `train` | HF dataset split (e.g., `validation`) |
| `--indexers` / `-i` | `all` | Comma-separated or `all` |
| `--model` | `ollama/nomic-embed-text` | Any LiteLLM model string |
| `--data-files` | none | Required for local CSV/JSON |
| `--no-cleanup` | — | Keep index files after run (default: auto-clean) |

### 4. Analyze and Recommend

After the benchmark table prints, give a concrete recommendation based on the user's constraint:

| Priority | Recommended indexer | Why |
|----------|-------------------|-----|
| Fastest queries | **Qdrant** or **FAISS** | Lowest query latency at scale |
| Lowest memory | **FAISS** (flat) or **LanceDB** | Minimal RAM overhead |
| Smallest index on disk | **LanceDB** | Column-oriented storage |
| Easiest to self-host | **Chroma** | Simple setup, good defaults |
| Production scale | **Qdrant** or **Milvus** | Built for distributed workloads |

Always tie the recommendation to the numbers from *their specific run*, not generic advice.

### 5. Cleanup

Index files are automatically removed after each run (default `--cleanup`). If the user wants to keep them for inspection, pass `--no-cleanup`. To manually purge any leftovers:
```bash
uv run embenx cleanup
```

## Safety

- Default `--max-docs 100` for first runs unless the user asks for more.
- Check that local file paths exist before running.
- Ensure Ollama is running if a local model is requested.
- If a backend isn't installed (e.g., milvus-lite), skip it or tell the user how to install it: `uv pip install milvus-lite`.

## Resources

- CLI help: `uv run embenx help` or `uv run embenx benchmark --help`
- Examples: `examples/ollama_benchmark.sh`
- Docs: `docs/index.html`
