Embenx Documentation 🚀
========================

Minimal, ultra-fast CLI for benchmarking vector indexing libraries.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   api
   contributing

Features
--------

- **Universal model support** — Ollama, OpenAI, Anthropic, and more via LiteLLM.
- **Any dataset** — Support for HuggingFace datasets, local files (CSV, JSON, JSONL, Parquet), and NumPy (.npy, .npz).
- **Advanced Indexers** — FAISS (Flat, IVF, HNSW, SQ8, PQ), ScaNN, HNSWLib, and more.
- **Quantization Support** — Benchmark Float16, Int8, and Product Quantization.
- **Database Support** — Benchmark pgvector (PostgreSQL), Elasticsearch, and Weaviate.
- **Custom Indexers** — Benchmark your own implementations with simple Python scripts.
- **Comprehensive Metrics** — Track Build Time, Query Latency, Index Size, and Memory Usage.

Quick Start
-----------

.. code-block:: bash

   # Install Embenx
   pip install embenx

   # Run a benchmark
   embenx benchmark --dataset squad --max-docs 100

Indices and tables
==================

Available Indexers:
-------------------

* **faiss**: FAISS (Flat)
* **faiss-ivf**: FAISS (Inverted File Index)
* **faiss-hnsw**: FAISS (Hierarchical Navigable Small World)
* **faiss-sq8**: FAISS (Scalar Quantizer 8-bit)
* **faiss-pq**: FAISS (Product Quantizer)
* **chroma**: ChromaDB
* **qdrant**: Qdrant
* **milvus**: Milvus
* **lance**: LanceDB
* **weaviate**: Weaviate
* **duckdb**: DuckDB
* **usearch**: USearch (Float32)
* **usearch-f16**: USearch (Float16)
* **usearch-i8**: USearch (Int8)
* **annoy**: Annoy (Approximate Nearest Neighbors Oh Yeah)
* **hnswlib**: HNSWLib (Hierarchical Navigable Small World)
* **scann**: ScaNN (Scalable Nearest Neighbors)
* **vespa**: Vespa
* **elasticsearch**: Elasticsearch
* **pgvector**: PostgreSQL (pgvector)
* **simple**: NumPy-based brute-force baseline

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
