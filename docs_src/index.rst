Embenx Documentation 🚀
========================

Python-native embedding retrieval layer.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   api
   contributing

Features
--------

- **Unified Collection API** — Table-like interface for vectors and metadata.
- **Metadata Filtering** — Native 'where' clause support for filtered retrieval.
- **Reranking Hooks** — Easily plug in Cross-Encoders or custom reranking logic.
- **Quantization Support** — SQ8, PQ, F16, and I8 indices for memory efficiency.
- **Universal model support** — Integrated LiteLLM for any embedding provider.
- **Portable Formats** — Native support for Parquet, NumPy (.npy/.npz), and FAISS (.index).
- **Multi-Backend** — 15+ backends including FAISS (IVF, HNSW), ScaNN, pgvector, USearch, and more.

Quick Start
-----------

.. code-block:: python

   from embenx import Collection
   col = Collection(dimension=768, indexer_type="faiss-hnsw")
   col.add(vectors, metadata)
   results = col.search(query, top_k=5)

Indices and tables
==================

Available Indexers:
-------------------

* **faiss**: FAISS (Flat, IVF, HNSW, SQ8, PQ)
* **scann**: ScaNN (Scalable Nearest Neighbors)
* **usearch**: USearch (F32, F16, I8)
* **pgvector**: PostgreSQL (pgvector)
* **lance**: LanceDB
* **chroma**: ChromaDB
* **qdrant**: Qdrant
* **milvus**: Milvus
* **annoy**: Annoy
* **hnswlib**: HNSWLib
* **weaviate**: Weaviate
* **duckdb**: DuckDB
* **elasticsearch**: Elasticsearch
* **vespa**: Vespa
* **simple**: NumPy-based brute-force baseline

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
