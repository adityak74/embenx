Usage Guide
===========

Embenx provides a high-level Python API and a powerful CLI for vector retrieval and benchmarking.

Python Library (Collection API)
-----------------------------

The ``Collection`` class is the primary entry point for using Embenx as a library.

Supported Indexer Types
^^^^^^^^^^^^^^^^^^^^^^

When initializing a ``Collection``, you can choose from the following ``indexer_type`` values:

* **FAISS Family**: ``faiss`` (Flat), ``faiss-ivf``, ``faiss-hnsw``, ``faiss-sq8``, ``faiss-pq``
* **Local ANN**: ``annoy``, ``hnswlib``, ``usearch``, ``usearch-f16``, ``usearch-i8``, ``chroma``, ``lance``, ``qdrant``, ``milvus``, ``scann``
* **Databases**: ``pgvector``, ``duckdb``, ``weaviate``, ``elasticsearch``, ``vespa``
* **Baselines**: ``simple`` (NumPy brute-force)

.. code-block:: python

   from embenx import Collection
   import numpy as np

   # 1. Initialize a collection
   # indexer_type can be: faiss, faiss-hnsw, faiss-sq8, usearch-f16, etc.
   col = Collection(dimension=768, indexer_type="faiss-hnsw")

   # 2. Add data
   vectors = np.random.rand(100, 768).astype('float32')
   metadata = [{"id": i, "category": "news"} for i in range(100)]
   col.add(vectors, metadata)

   # 3. Search with filtering
   results = col.search(
       query=vectors[0], 
       top_k=5, 
       where={"category": "news"}
   )

   # 4. Search with custom reranking
   def my_reranker(query, results):
       return sorted(results, key=lambda x: x[0]['id'], reverse=True)

   results = col.search(vectors[0], reranker=my_reranker)

   # 5. Benchmark multiple indexers on live data
   # This compares backends directly on your collection's current state
   col.benchmark(indexers=["faiss", "usearch", "simple"])

Hybrid Search (Dense + Sparse)
----------------------------

Embenx supports combining dense vector search with sparse BM25 retrieval.

.. code-block:: python

   # Initialize with both indexers
   col = Collection(
       dimension=768, 
       indexer_type="faiss-hnsw", 
       sparse_indexer_type="bm25"
   )

   # Add data (ensure metadata has a 'text' field for BM25)
   col.add(vectors, metadata=[{"text": "content here", "id": 1}])

   # Search using Reciprocal Rank Fusion (RRF)
   results = col.hybrid_search(
       query_vector=my_vector,
       query_text="search keywords",
       top_k=5,
       dense_weight=0.7,
       sparse_weight=0.3
   )

Benchmark CLI
-------------

The primary CLI command is ``benchmark``:

.. code-block:: bash

   embenx benchmark --dataset <dataset_name> [options]

Options:

* ``--dataset`` / ``-d``: HuggingFace dataset name or format (csv, json, parquet). You can also pass a local file path directly.
* ``--max-docs`` / ``-m``: Maximum documents to index.
* ``--indexers`` / ``-i``: Comma-separated list of indexers to test.
* ``--model``: LiteLLM model string (e.g., ``ollama/nomic-embed-text``).
* ``--custom-indexer``: Path to a custom indexer Python script.

Environment Setup
-----------------

Check your environment before running:

.. code-block:: bash

   embenx setup --pull

Local Datasets
--------------

Embenx supports local CSV, JSON, Parquet, and NumPy files. You can pass the path directly to the ``--dataset`` flag:

.. code-block:: bash

   # Using a direct path to a Parquet file
   embenx benchmark --dataset ./my_data.parquet --text-column content

   # Using a direct path to a NumPy file
   embenx benchmark --dataset ./embeddings.npy

Custom Indexers
---------------

You can create a custom indexer by inheriting from ``BaseIndexer``:

.. code-block:: python

   from embenx import BaseIndexer

   class MyIndexer(BaseIndexer):
       def build_index(self, embeddings, metadata):
           # build logic
           pass

       def search(self, query_embedding, top_k=5):
           # search logic
           return []

       def get_size(self):
           return 0

Run your custom indexer with the ``--custom-indexer`` flag:

.. code-block:: bash

   embenx benchmark --custom-indexer ./my_indexer.py --indexers myindexername
