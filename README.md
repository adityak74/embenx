<div align="center">

<h1>Embenx 🚀</h1>

<p>
  <strong>Universal embedding retrieval toolkit & benchmark.</strong><br/>
  Search, filter, and rerank across 15+ vector backends (FAISS, ScaNN, pgvector, etc.) with a unified Python API and CLI.
</p>

<p>
  <a href="https://github.com/adityak74/embenx/stargazers"><img src="https://img.shields.io/github/stars/adityak74/embenx?style=flat-square&color=yellow" alt="Stars"/></a>
  <a href="https://github.com/adityak74/embenx/issues"><img src="https://img.shields.io/github/issues/adityak74/embenx?style=flat-square" alt="Issues"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="MIT License"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square" alt="Python 3.9+"/></a>
  <a href="https://adityak74.github.io/embenx/"><img src="https://img.shields.io/badge/docs-live-brightgreen?style=flat-square" alt="Docs"/></a>
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

Embenx is a Python-native retrieval library that sits between raw vector indices and full-blown vector databases. It provides a high-level `Collection` API for managing embeddings and metadata, supporting advanced features like **filtering**, **reranking**, and **quantization** across 15+ backends.

## Library Usage

```python
from embenx import Collection

# 1. Initialize a collection
col = Collection(dimension=768, indexer_type="faiss-hnsw")

# 2. Add data
col.add(
    vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    metadata=[{"category": "AI", "id": 1}, {"category": "Tools", "id": 2}]
)

# 3. Search with filtering
results = col.search(
    query=[0.1, 0.2, ...],
    top_k=5,
    where={"category": "AI"}
)

# 4. Save/Load
col.to_parquet("my_collection.parquet")
new_col = Collection.from_parquet("my_collection.parquet")

# 5. Benchmark multiple indexers on your data
col.benchmark(indexers=["faiss", "usearch", "hnswlib"])
```

## Agentic Memory (MCP)

Embenx ships with a built-in **Model Context Protocol (MCP)** server. This allows AI agents (like Claude Desktop) to use Embenx collections as their own long-term memory.

### 1. Start the server
```bash
embenx mcp-start
```

## Visual Explorer

Embenx provides a built-in web UI to visualize your vector collections and metadata.

```bash
embenx explorer
```

### 2. Configure Claude Desktop
Add this to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "embenx": {
      "command": "uv",
      "args": ["--directory", "/path/to/embenx", "run", "embenx", "mcp-start"]
    }
  }
}
```

## Features
...
- **Unified Collection API** — Table-like interface for vectors and metadata.
- **ClusterKV Optimization** — Semantic clustering for high-throughput retrieval (arXiv:2412.03213).
- **KV Cache Offloading (RA-KVC)** — Store and retrieve high-dimensional LLM activations using `safetensors`.
- **SSM State Hydration** — Persist and prime hidden states ($h_0$) for State Space Models (Mamba-2).
- **Trajectory Retrieval** — Search for similar state/action sequences for World Models.
- **Visual Explorer** — Built-in web UI to visualize vector clusters and metadata.
- **Agentic Memory (MCP)** — Native Model Context Protocol support for AI agents.
- **Hybrid Search** — Combine dense vectors with sparse BM25 retrieval using RRF.

- **Recall Evaluation** — Built-in tools to measure ANN accuracy against exact search.
- **Library-Native Benchmarking** — Compare performance directly from Python code.
- **Metadata Filtering** — Native 'where' clause support for filtered retrieval.
- **Reranking Hooks** — Easily plug in Cross-Encoders or custom reranking logic.
- **Quantization Support** — SQ8, PQ, F16, and I8 indices for memory efficiency.
- **Universal model support** — Integrated LiteLLM for any embedding provider.
- **Portable Formats** — Native support for Parquet, NumPy (.npy/.npz), and FAISS (.index).
- **Multi-Backend** — 15+ backends including FAISS (IVF, HNSW), ScaNN, pgvector, USearch, and more.

## Supported Indexers

| Indexer | Family | Best For |
| :--- | :--- | :--- |
| `faiss` | HNSW, IVF, Flat | Production-grade local search |
| `scann` | Tree-AH | State-of-the-art speed/recall (Linux) |
| `usearch` | HNSW | High-performance C++, low latency |
| `pgvector` | Postgres | Embeddings next to relational data |
| `lancedb` | Columnar | Large disk-based datasets |
| `simple` | NumPy | Exact search baseline |

## Installation

```bash
pip install embenx
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for our journey towards hybrid search and production-grade retrieval.

## License

Distributed under the **MIT License**.

---

<div align="center">
  Built with ❤️ for the AI engineering community by <a href="https://github.com/adityak74">adityak74</a>
</div>
