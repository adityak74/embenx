<div align="center">

<h1>Embenx 🚀</h1>

<p>
  <strong>Minimal, ultra-fast CLI for benchmarking vector indexing libraries.</strong><br/>
  Compare FAISS (Flat, IVF, HNSW), Chroma, Qdrant, Milvus, LanceDB, Weaviate, DuckDB, USearch, Annoy, HNSWLib, ScaNN, Vespa, Elasticsearch, and pgvector with any dataset.
</p>

<p>
  <a href="https://github.com/adityak74/embenx/stargazers"><img src="https://img.shields.io/github/stars/adityak74/embenx?style=flat-square&color=yellow" alt="Stars"/></a>
  <a href="https://github.com/adityak74/embenx/issues"><img src="https://img.shields.io/github/issues/adityak74/embenx?style=flat-square" alt="Issues"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="MIT License"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square" alt="Python 3.9+"/></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-supported-orange.svg?style=flat-square" alt="Ollama Support"/></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/uv-ready-purple.svg?style=flat-square" alt="uv ready"/></a>
</p>

<p>
  <a href="https://adityak74.github.io/embenx/">Documentation</a> ·
  <a href="https://github.com/adityak74/embenx/issues">Report Bug</a> ·
  <a href="https://github.com/adityak74/embenx/issues">Request Feature</a>
</p>

</div>

---

## What is Embenx?

Embenx is a single-command benchmarking tool for vector search backends. Point it at any HuggingFace dataset or local file, pick your embedding model, and get a side-by-side comparison of build time, query latency, index size, and memory usage across all major vector libraries.

```
Indexer   Build(s)  Query(ms)  Index(KB)  Memory(MB)
--------  --------  ---------  ---------  ----------
faiss         0.42       0.31        580        18.4
faiss-hnsw    0.55       0.08        620        22.1
faiss-ivf     1.20       0.12        590        45.0
chroma        1.10       3.21       1200        41.0
qdrant        2.31       1.94       4800        64.2
lance         0.55       2.10        310        22.0
milvus        3.74       2.60       2400        87.0
pgvector      0.85       1.42        950        12.0
```

## Features

- **Universal model support** — Ollama (local, free), OpenAI, Anthropic, and any LiteLLM provider
- **Any dataset** — HuggingFace datasets, local CSV / JSON / JSONL / Parquet, and NumPy (.npy, .npz)
- **Advanced Indexers** — FAISS (Flat, IVF, HNSW), ScaNN, HNSWLib, and more
- **Database Support** — Benchmark pgvector (PostgreSQL), Elasticsearch, and Weaviate
- **Custom indexer scripts** — benchmark your own implementations by providing a script path
- **Four metrics** — build time, query latency, index footprint, memory overhead
- **Environment setup** — `embenx setup` checks all dependencies in one command
- **AI-agent ready** — ships with `embenx init-skill` to generate a `SKILL.md` for AI agents

## Installation

**Recommended (uv):**
```bash
git clone https://github.com/adityak74/embenx.git
cd embenx
uv sync
```

**pip:**
```bash
pip install .
```

## Quick Start

### 1. Check your environment
```bash
uv run embenx setup
```
This verifies all indexers are installed and your Ollama model is ready. Pass `--pull` to auto-download a missing model.

### 2. Run a benchmark
```bash
uv run embenx benchmark --dataset squad --max-docs 100 --model ollama/nomic-embed-text
```

### 3. Benchmark local data
```bash
# Direct path to Parquet, CSV, or NumPy
uv run embenx benchmark --dataset ./my_data.parquet --text-column content
uv run embenx benchmark --dataset ./embeddings.npy
```

### 4. Custom Indexers
Embenx allows you to benchmark your own indexing implementations. Create a Python script that inherits from `BaseIndexer` and pass it to the CLI.

```python
# my_indexer.py
from indexers.base import BaseIndexer

class MyCustomIndexer(BaseIndexer):
    def build_index(self, embeddings, metadata):
        # Your logic here
        pass

    def search(self, query_embedding, top_k=5):
        # Your logic here
        return []

    def get_size(self):
        return 0
```

Run it with:
```bash
uv run embenx benchmark --custom-indexer ./my_indexer.py --indexers mycustomindexer
```

## CLI Reference

### `embenx benchmark`

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--dataset` / `-d` | HuggingFace dataset name or local path (csv, json, parquet, npy, index) | *required* |
| `--text-column` / `-c` | Column to embed and index | `text` |
| `--split` / `-s` | HuggingFace dataset split | `train` |
| `--max-docs` / `-m` | Number of documents to index | `1000` |
| `--indexers` / `-i` | Comma-separated list or `all` | `all` |
| `--model` | LiteLLM model string | `ollama/nomic-embed-text` |
| `--data-files` | Path to local file(s) | — |
| `--no-cleanup` | Keep index files on disk after the run | — |
| `--custom-indexer` | Path to a Python script with a custom indexer class | — |

## Output Metrics

| Metric | What it measures |
| :--- | :--- |
| **Build Time (s)** | Time to embed all documents and construct the index |
| **Query Time (ms/query)** | Mean search latency across 10 test queries |
| **Index Size (KB)** | Disk or memory footprint of the final index |
| **Memory Added (MB)** | Process RAM delta during the indexing phase |

## License

Distributed under the **MIT License** — free to use, modify, and distribute with attribution.

See [`LICENSE`](LICENSE) for the full license text.

---

<div align="center">
  Built with ❤️ for the AI engineering community by <a href="https://github.com/adityak74">adityak74</a>
</div>
