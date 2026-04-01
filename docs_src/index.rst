Embenx Documentation 🚀
========================

**Universal embedding retrieval toolkit & agentic memory layer.**

Embenx is a high-performance Python library designed for the 2026 AI ecosystem. It bridges the gap between raw vector indices and full-blown vector databases, providing a unified API for 15+ backends, advanced filtering, hybrid search, and specialized memory structures for AI agents.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   cli
   api
   contributing

Key Capabilities
----------------

* **Unified Collection API**: A table-like interface for managing vectors and metadata seamlessly.
* **15+ Vector Backends**: Support for FAISS, ScaNN, USearch, pgvector, LanceDB, Milvus, Qdrant, and more.
* **Agentic Memory (MCP)**: Built-in Model Context Protocol server for instant integration with Claude, GPT-5, and autonomous agents.
* **Advanced Retrieval**: Native support for Matryoshka embeddings (dimension truncation), RRF-based Hybrid Search, and Cross-Encoder reranking.
* **Research-Driven Optimizations**: Implementation of state-of-the-art algorithms like ClusterKV, TurboQuant (1-bit quantization), and Echo (Temporal Memory).
* **Visual & DevTools**: Built-in 3D HNSW Graph Visualizer, RAG Playground, and one-click production export.
* **Multimodal Support**: Native handling of image embeddings (CLIP) and cross-modal retrieval.

Quick Start
-----------

Install Embenx:

.. code-block:: bash

   pip install embenx

Simple semantic search:

.. code-block:: python

   from embenx import Collection
   import numpy as np

   # Initialize a collection with FAISS-HNSW
   col = Collection(dimension=768, indexer_type="faiss-hnsw")

   # Add data with metadata
   vectors = np.random.rand(100, 768).astype('float32')
   metadata = [{"id": i, "text": f"Document {i}", "category": "news"} for i in range(100)]
   col.add(vectors, metadata)

   # Search with metadata filtering
   results = col.search(query_vector, top_k=5, where={"category": "news"})

   for meta, dist in results:
       print(f"Found: {meta['text']} (Distance: {dist:.4f})")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
