Usage Guide
===========

Embenx is designed to be simple for prototyping yet robust enough for research-grade agentic memory. This guide covers core retrieval and serialization.

Core Retrieval
--------------

The primary interface is the ``Collection`` class. It provides a table-like abstraction for vectors and metadata.

.. code-block:: python

   from embenx import Collection
   import numpy as np

   # 1. Initialize with a specific backend
   # Options: 'faiss-hnsw', 'scann', 'usearch', 'pgvector', 'duckdb', etc.
   col = Collection(dimension=768, indexer_type="faiss-hnsw")

   # 2. Add data
   # Vectors can be numpy arrays or lists
   vectors = np.random.rand(100, 768).astype('float32')
   metadata = [{"id": i, "text": f"Document {i}", "tag": "test"} for i in range(100)]
   col.add(vectors, metadata)

   # 3. Basic Search
   # Returns a list of (metadata, distance) tuples
   results = col.search(query_vector, top_k=5)

   # 4. Metadata Filtering
   # Supports exact match dictionary filters across any indexed field
   results = col.search(query_vector, top_k=5, where={"tag": "test"})

   # 5. Serialization
   # Saves to a portable Parquet file containing both vectors and metadata
   col.to_parquet("my_memory.parquet")
   
   # Load back
   new_col = Collection.from_parquet("my_memory.parquet")

Advanced Retrieval Features
--------------------------

Matryoshka Truncation
~~~~~~~~~~~~~~~~~~~~~

If you are using Matryoshka Representation Learning (MRL) models, you can truncate dimensions for 10x faster retrieval with minimal accuracy loss.

.. code-block:: python

   # Define a collection that truncates 768-dim embeddings to 128
   col = Collection(dimension=768, truncate_dim=128)
   
   # Input vectors are still expected to be 768-dim; truncation happens internally
   col.add(full_vectors, metadata)
   results = col.search(full_query_vector)

Hybrid Search (Dense + Sparse)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine semantic vector search with keyword-based BM25 retrieval using Reciprocal Rank Fusion (RRF).

.. code-block:: python

   # Initialize with a sparse indexer
   col = Collection(dimension=768, sparse_indexer_type="bm25")
   
   # Perform hybrid search
   results = col.hybrid_search(
       query_vector=q_vec,
       query_text="fox",
       dense_weight=0.5,
       sparse_weight=0.5
   )

Reranking
~~~~~~~~~

Improve precision by re-scoring top candidates with a Cross-Encoder or FlashRank.

.. code-block:: python

   from embenx.rerank import RerankHandler
   
   # Use FlashRank (CPU-optimized)
   ranker = RerankHandler(model_name="ms-marco-TinyBERT-L-2-v2", model_type="flashrank")
   
   # Search with reranking hook
   results = col.search(query_vector, top_k=5, reranker=ranker, query_text="My original question")

Evaluation & Benchmarking
-------------------------

Embenx makes it easy to measure the performance of different indexers on your own data.

.. code-block:: python

   # Measure Recall@10 against an exact search baseline
   metrics = col.evaluate(indexer_type="faiss-hnsw", top_k=10)
   print(f"Recall: {metrics['recall']}, Latency: {metrics['latency_ms']}ms")

   # Benchmark multiple indexers side-by-side
   col.benchmark(indexers=["faiss", "usearch", "hnswlib"])
