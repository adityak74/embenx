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
- **Any dataset** — Support for HuggingFace datasets and local files (CSV, JSON, JSONL).
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

* **faiss**: FAISS (Facebook AI Similarity Search)
* **chroma**: ChromaDB
* **qdrant**: Qdrant
* **milvus**: Milvus
* **lance**: LanceDB
* **weaviate**: Weaviate
* **duckdb**: DuckDB
* **usearch**: USearch
* **annoy**: Annoy (Approximate Nearest Neighbors Oh Yeah)
* **hnswlib**: HNSWLib (Hierarchical Navigable Small World)
* **scann**: ScaNN (Scalable Nearest Neighbors)
* **vespa**: Vespa
* **elasticsearch**: Elasticsearch
* **simple**: NumPy-based brute-force baseline

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
