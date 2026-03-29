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
- **Any dataset** — HuggingFace datasets, local files (CSV, JSON, JSONL, Parquet), and NumPy (.npy, .npz).
- **Advanced Indexers** — FAISS (Flat, IVF, HNSW), ScaNN, HNSWLib, and more.
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
* **chroma**: ChromaDB
* **qdrant**: Qdrant
* **milvus**: Milvus
* **lance**: LanceDB
* **weaviate**: Weaviate
* **duckdb**: DuckDB
* **usearch**: USearch
* **annoy**: Annoy
* **hnswlib**: HNSWLib
* **scann**: ScaNN
* **vespa**: Vespa
* **elasticsearch**: Elasticsearch
* **pgvector**: PostgreSQL (pgvector)
* **simple**: NumPy-based brute-force baseline

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
