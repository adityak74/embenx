<!-- generated-by: gsd-doc-writer -->
Usage Guide
===========

Embenx is designed to be simple for prototyping yet robust enough for research-grade agentic memory. This guide covers core retrieval, serialization, and advanced search patterns.

Creating a Collection
---------------------

The primary interface is the ``Collection`` class. You can initialize it with a specific backend and dimension.

.. code-block:: python

   from embenx import Collection

   # Initialize a collection for 768-dimensional vectors using FAISS HNSW
   col = Collection(
       name="my_collection", 
       dimension=768, 
       indexer_type="faiss-hnsw"
   )

Supported Backends:
- **Local/In-Memory**: ``faiss``, ``faiss-hnsw``, ``scann``, ``usearch``, ``hnswlib``, ``annoy``, ``simple``
- **Native DBs**: ``duckdb``, ``pgvector``, ``lance``, ``chroma``
- **Managed/Distributed**: ``milvus``, ``qdrant``, ``weaviate``, ``opensearch``, ``elasticsearch``, ``vespa``

Inserting Data
--------------

You can insert vectors and metadata using the ``add`` method. Vectors can be NumPy arrays or lists of floats.

.. code-block:: python

   import numpy as np

   # Generate some dummy data
   vectors = np.random.rand(100, 768).astype('float32')
   metadata = [{"id": i, "text": f"Document {i}", "category": "news"} for i in range(100)]

   # Insert data
   col.add(vectors, metadata)

For large-scale ingestion, use ``add_batch`` to manage memory and show progress:

.. code-block:: python

   col.add_batch(
       large_vectors, 
       large_metadata, 
       batch_size=1000, 
       show_progress=True
   )

Performing Search
-----------------

Search for the nearest neighbors using a query vector.

.. code-block:: python

   # Search for the top 5 results
   query_vector = np.random.rand(768).astype('float32')
   results = col.search(query_vector, top_k=5)

   # results is a list of (metadata, distance) tuples
   for meta, distance in results:
       print(f"ID: {meta['id']}, Distance: {distance}")

Metadata Filtering
~~~~~~~~~~~~~~~~~~

You can filter results based on metadata fields:

.. code-block:: python

   # Filter by category
   results = col.search(query_vector, top_k=5, where={"category": "news"})

OpenSearch Example
------------------

To use OpenSearch as a backend, ensure you have an instance running and specify ``indexer_type="opensearch"``.

.. code-block:: python

   # Ensure opensearch-py is installed: pip install opensearch-py
   # Default URL: http://localhost:9200 (can be overridden via OPENSEARCH_URL env var)
   
   col_os = Collection(
       name="opensearch_collection",
       dimension=768,
       indexer_type="opensearch"
   )

   # Add and search work the same way
   # Vectors and metadata are stored in OpenSearch using the k-NN plugin
   col_os.add(vectors, metadata)
   results = col_os.search(query_vector, top_k=3)

Hybrid Search (Dense + Sparse)
------------------------------

Combine semantic vector search with keyword-based BM25 retrieval using Reciprocal Rank Fusion (RRF).

.. code-block:: python

   # Initialize with both a dense and a sparse indexer
   col_hybrid = Collection(
       dimension=768, 
       indexer_type="faiss-hnsw",
       sparse_indexer_type="bm25"
   )
   
   # Add data (BM25 will index the 'text' field in metadata)
   metadata = [{"id": i, "text": f"This is document {i}"} for i in range(100)]
   col_hybrid.add(vectors, metadata)

   # Perform hybrid search
   # Requires both a query vector and the query text
   results = col_hybrid.hybrid_search(
       query_vector=query_vector,
       query_text="document search query",
       dense_weight=0.5,
       sparse_weight=0.5
   )

Advanced Search Patterns
------------------------

Image Search
~~~~~~~~~~~~

Native support for image retrieval using CLIP embeddings.

.. code-block:: python

   # Search using a local image path
   results = col.search_image("path/to/my_image.png", top_k=5)

Trajectory Search
~~~~~~~~~~~~~~~~~

Search for similar sequences of states/actions by pooling trajectories into a single vector.

.. code-block:: python

   # pooling can be 'mean' or 'max'
   results = col.search_trajectory(my_state_sequence, pooling="mean")

Matryoshka Truncation
~~~~~~~~~~~~~~~~~~~~~

If you are using Matryoshka Representation Learning (MRL) models, you can truncate dimensions for 10x faster retrieval with minimal accuracy loss.

.. code-block:: python

   # Define a collection that truncates 768-dim embeddings to 128
   col = Collection(dimension=768, truncate_dim=128)
   
   # Input vectors are still expected to be 768-dim; truncation happens internally
   col.add(full_vectors, metadata)
   results = col.search(full_query_vector)

Reranking
~~~~~~~~~

Improve precision by re-scoring top candidates with a Cross-Encoder or FlashRank.

.. code-block:: python

   from embenx.rerank import RerankHandler
   
   # Use FlashRank (CPU-optimized)
   ranker = RerankHandler(model_name="ms-marco-TinyBERT-L-2-v2", model_type="flashrank")
   
   # Search with reranking hook
   results = col.search(
       query_vector, 
       top_k=5, 
       reranker=ranker, 
       query_text="My original question"
   )

Serialization
-------------

Save and load collections using Parquet files.

.. code-block:: python

   # Save to Parquet
   col.to_parquet("my_memory.parquet")
   
   # Load from Parquet
   new_col = Collection.from_parquet("my_memory.parquet")

Evaluation & Benchmarking
-------------------------

Embenx makes it easy to measure the performance of different indexers.

.. code-block:: python

   # Measure Recall@10 against an exact search baseline
   metrics = col.evaluate(indexer_type="faiss-hnsw", top_k=10)
   print(f"Recall: {metrics['recall']}, Latency: {metrics['latency_ms']}ms")

   # Benchmark multiple indexers side-by-side
   col.benchmark(indexers=["faiss", "usearch", "hnswlib"])

Synthetic Data Generation
-------------------------

Generate high-quality synthetic query-document pairs from your collections using LLMs.

.. code-block:: python

   # Generate queries using LiteLLM (v1.83.0+)
   results = col.generate_synthetic_queries(
       text_key="text",
       n_queries_per_doc=2,
       num_docs=100,
       model="gpt-4o-mini"
   )
