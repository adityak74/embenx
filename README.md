<div align="center">

<img src="embenx.png" alt="Embenx Logo" width="200"/>

<h1>Embenx — Agentic Memory Layer for Python AI Agents 🚀</h1>

<p>
  <strong>The Agentic Memory Layer & Universal Retrieval Toolkit.</strong><br/>
  Synthetic data generation, 15+ vector backends, hybrid search, and MCP native memory for AI agents.
</p>

<p>
  <a href="https://github.com/adityak74/embenx/stargazers"><img src="https://img.shields.io/github/stars/adityak74/embenx?style=flat-square&color=yellow" alt="Stars"/></a>
  <a href="https://github.com/adityak74/embenx/issues"><img src="https://img.shields.io/github/issues/adityak74/embenx?style=flat-square" alt="Issues"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="MIT License"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square" alt="Python 3.11+"/></a>
  <a href="https://adityak74.github.io/embenx/"><img src="https://img.shields.io/badge/docs-live-brightgreen?style=flat-square" alt="Docs"/></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/uv-ready-purple.svg?style=flat-square" alt="uv ready"/></a>
</p>

<p>
  <a href="https://pypi.org/project/embenx/"><img src="https://img.shields.io/pypi/v/embenx?style=flat-square&color=blue" alt="PyPI Version"/></a>
  <a href="https://pypi.org/project/embenx/"><img src="https://img.shields.io/pypi/dw/embenx?style=flat-square&color=brightgreen" alt="PyPI Weekly Downloads"/></a>
</p>

<p>
  <strong><a href="https://adityak74.github.io/embenx/">📖 Read the Docs</a></strong> &nbsp;·&nbsp;
  <a href="https://adityak74.github.io/embenx/">Explore the Visual UI</a> &nbsp;·&nbsp;
  <a href="https://github.com/adityak74/embenx/issues">Report Bug</a> &nbsp;·&nbsp;
  <a href="https://github.com/adityak74/embenx/issues">Request Feature</a>
</p>

</div>

---

## What is Embenx?

Embenx is a Python-native retrieval library that sits between raw vector indices and full-blown vector databases. It provides a high-level `Collection` API for managing embeddings and metadata, supporting advanced features like **filtering**, **reranking**, and **quantization** across 15+ backends.

## Quickstart

Get up and running in 60 seconds.

**Step 1 — Install**
```bash
pip install embenx
```

**Step 2 — Create a collection and add embeddings**
```python
import numpy as np
from embenx import Collection

# 768-dim FAISS-HNSW index (in-memory, no extra config needed)
col = Collection(dimension=768, indexer_type="faiss-hnsw")

vectors = np.random.rand(10, 768).astype("float32")
metadata = [{"id": i, "text": f"Document {i}"} for i in range(10)]
col.add(vectors, metadata)
```

**Step 3 — Search**
```python
query = np.random.rand(768).astype("float32")
results = col.search(query, top_k=3)

for meta, dist in results:
    print(f"{meta['text']}  (distance: {dist:.4f})")
```

> For filtering, reranking, hybrid search, and production export, see [Library Usage](#library-usage) or the [full docs](https://adityak74.github.io/embenx/).

---

## Library Usage

```python
from embenx import Collection

# 1. Initialize a collection
col = Collection(dimension=768, indexer_type="faiss-hnsw")

# 2. Add data
# Supports incremental O(1) additions and bulk 'add_batch'
col.add(
    vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    metadata=[{"category": "AI", "id": 1, "text": "The quick brown fox"}]
)

# Bulk batch ingestion with progress bar
col.add_batch(large_vectors, batch_size=1000, show_progress=True)

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

<!-- TODO: Add HNSW visualizer GIF here -->

```bash
embenx explorer
```

> **[Open the Explorer UI →](https://adityak74.github.io/embenx/visual.html)**  
> Launch the visual dashboard, explore HNSW graph layers, run a RAG Playground session, and inspect cluster distributions — all from your browser.

## Synthetic Data Generation 🧪

Generate high-quality query-document pairs to train or evaluate your retrieval pipelines. Embenx supports **LiteLLM** (for 100+ providers like OpenAI, Anthropic, Gemini) and local **Ollama** models.

```python
from embenx import Collection

col = Collection.load("my_collection")

# Generate 2 synthetic queries for each of the first 100 documents
results = col.generate_synthetic_queries(
    n_queries_per_doc=2,
    num_docs=100,
    model="gpt-4o-mini",  # Or "ollama/llama3"
    output_path="eval_data.jsonl"
)
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

Embenx registers **22 indexer keys** across **12 backend families** out of the box.

| Indexer Key | Family / Algorithm | Best For |
| :--- | :--- | :--- |
| `faiss` | FAISS Flat | Exact baseline (GPU-ready) |
| `faiss-ivf` | FAISS IVF | Large-scale approximate search |
| `faiss-hnsw` | FAISS HNSW | High-recall in-memory search |
| `faiss-sq8` | FAISS SQ8 | Quantized, memory-efficient search |
| `faiss-pq` | FAISS PQ | Ultra-compressed approximate search |
| `scann` | ScaNN Tree-AH | State-of-the-art speed/recall (Linux) |
| `usearch` | USearch HNSW (f32) | High-performance C++, low latency |
| `usearch-f16` | USearch HNSW (f16) | Half-precision, memory-efficient |
| `usearch-i8` | USearch HNSW (i8) | Integer quantized, minimal RAM |
| `hnswlib` | HNSWLib | Pure HNSW, easy to tune |
| `annoy` | Annoy (Random Projections) | Read-heavy / static datasets |
| `pgvector` | PostgreSQL pgvector | Embeddings next to relational data |
| `lance` | LanceDB Columnar | Large disk-based datasets |
| `milvus` | Milvus Cluster | Distributed production workloads |
| `qdrant` | Qdrant | Filtered vector search at scale |
| `chroma` | ChromaDB | Lightweight local development |
| `weaviate` | Weaviate | Multi-tenant, schema-driven search |
| `duckdb` | DuckDB | Analytical + vector hybrid queries |
| `elasticsearch` | Elasticsearch | Full-text + vector search combined |
| `opensearch` | OpenSearch | Native k-NN vector search |
| `vespa` | Vespa | Real-time ranking & serving |
| `bm25` | BM25 (sparse) | Keyword / sparse retrieval baseline |
| `simple` | NumPy Exact | Exact search, zero dependencies |

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
