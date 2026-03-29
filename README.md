<div align="center">

<h1>Embenx 🚀</h1>

<p>
  <strong>Python-native embedding retrieval layer.</strong><br/>
  High-performance vector search, filtering, and reranking with zero database overhead.
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

Embenx is a Python-native retrieval library that sits between raw vector indices and full-blown vector databases. It provides a high-level `Collection` API for managing embeddings and metadata, supporting advanced features like **filtering**, **reranking**, and **quantization** with multiple backends (FAISS, LanceDB, Qdrant, etc.).

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
```

## Features

- **Unified Collection API** — Table-like interface for vectors and metadata.
- **Metadata Filtering** — Native 'where' clause support for filtered retrieval.
- **Reranking Hooks** — Easily plug in Cross-Encoders or custom reranking logic.
- **Quantization Support** — Benchmark and use SQ8, PQ, F16, and I8 indices.
- **Universal model support** — Integrated LiteLLM for any embedding provider.
- **Portable Formats** — Native support for Parquet, NumPy (.npy/.npz), and FAISS (.index).
- **Multi-Backend** — FAISS, Chroma, Qdrant, Milvus, LanceDB, Weaviate, DuckDB, USearch, and more.

## Installation

```bash
pip install embenx
```

## CLI Benchmarking

Embenx still ships with its powerful benchmarking CLI:

```bash
# Benchmark multiple indexers on a HF dataset
embenx benchmark --dataset squad --max-docs 100 --indexers faiss,qdrant,lance
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for our journey towards hybrid search and production-grade retrieval.

## License

Distributed under the **MIT License**.

---

<div align="center">
  Built with ❤️ for the AI engineering community by <a href="https://github.com/adityak74">adityak74</a>
</div>
