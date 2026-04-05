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
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square" alt="Python 3.10+"/></a>
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
    metadata=[{"category": "AI", "id": 1, "text": "The quick brown fox"}]
)

# 3. Search with filtering
results = col.search(
    query=[0.1, 0.2, ...],
    top_k=5,
    where={"category": "AI"}
)

# 4. Export to production
col.export_to_production(backend="qdrant", connection_url="http://localhost:6333")

# 5. Generate synthetic data (training/eval)
# Supports LiteLLM (v1.83.0+) and local Ollama
pairs = col.generate_synthetic_queries(
    n_queries_per_doc=2,
    output_path="synthetic_data.jsonl"
)
```

## Agentic Memory (MCP)

Embenx ships with a built-in **Model Context Protocol (MCP)** server. This allows AI agents (like Claude Desktop) to use Embenx collections as their own long-term memory.

### 1. Start the server
```bash
embenx mcp-start
```

## Visual Explorer

Embenx provides a built-in web UI to visualize your vector collections, including an interactive **HNSW Graph Visualizer** and a **RAG Playground**.

```bash
embenx explorer
```

## Features

- **Synthetic Data Generation** — Create high-quality query-document pairs using LLMs for training and evaluation.
- **Multimodal Support** — Native support for image embeddings (CLIP).
- **RAG Playground** — Test retrieval quality with an integrated LLM chat loop.
- **HNSW Graph Visualizer** — Interactive 3D visualization of navigation layers.
- **Export to Production** — One-click migration to Qdrant or Milvus clusters.
- **Unified Collection API** — Table-like interface for vectors and metadata.
- **Retrieval Zoo** — Instant access to pre-indexed collections (SQuAD, MS-MARCO, etc.).
- **Agentic Memory (MCP)** — Native Model Context Protocol support for AI agents.
- **Self-Healing Retrieval** — Integrated feedback loops to automatically improve ranking accuracy.
- **Temporal Memory (Echo)** — Recency-biased retrieval and time-window filtering (arXiv:2502.16090).
- **Spatial Memory (ESWM)** — Neuroscience-inspired spatial cognitive maps for navigation (ICLR 2026).
- **TurboQuant Compression** — 1-bit sign-based quantization for activation tensors (arXiv:2504.19874).
- **ClusterKV Optimization** — Semantic clustering for high-throughput retrieval (arXiv:2412.03213).
- **Hybrid Search** — Combine dense vectors with sparse BM25 retrieval using RRF.
- **KV Cache Offloading (RA-KVC)** — Store and retrieve high-dimensional LLM activations using `safetensors`.
- **SSM State Hydration** — Persist and prime hidden states ($h_0$) for State Space Models (Mamba-2).
- **Trajectory Retrieval** — Search for similar state/action sequences for World Models.
- **Visual Explorer** — Built-in web UI to visualize vector clusters and metadata.
- **Universal model support** — Integrated LiteLLM for any embedding provider.
- **Portable Formats** — Native support for Parquet, NumPy (.npy/.npz), and FAISS (.index).

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

See [ROADMAP.md](ROADMAP.md) for our journey towards production-grade agentic retrieval.

## License

Distributed under the **MIT License**.

---

<div align="center">
  Built with ❤️ for the AI engineering community by <a href="https://github.com/adityak74">adityak74</a>
</div>
