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

   # 6. Evaluate recall and latency
   # Compares an ANN indexer against an exact search baseline
   metrics = col.evaluate(indexer_type="faiss-hnsw", top_k=10)
   print(f"Recall: {metrics['recall']}, Latency: {metrics['latency_ms']}ms")

Trajectory Retrieval
-------------------

Embenx supports searching for similar sequences of vectors, which is useful for World Models and robotic state trajectories.

.. code-block:: python

   # Define a sequence of vectors
   trajectory = [
       [0.1, 0.2, 0.3, 0.4],
       [0.2, 0.3, 0.4, 0.5]
   ]
   
   # Search using mean-pooling (default)
   results = col.search_trajectory(trajectory, top_k=5, pooling="mean")

KV Cache Offloading (RA-KVC)
---------------------------

Embenx provides a specialized ``CacheCollection`` for storing and retrieving high-dimensional LLM activations (KV cache pairs).

.. code-block:: python

   from embenx.core import CacheCollection
   
   col = CacheCollection(name="my_cache", dimension=128)
   
   # Add embeddings and associated activations
   # activations is a dict of numpy arrays
   col.add_cache(vectors, activations={"k": k_tensors, "v": v_tensors}, quantize=True)
   
   # Retrieve later
   results = col.search(query_vector)
   cached_kv = col.get_cache(results[0][0])

SSM State Hydration
------------------

For architectures like Mamba-2 or Jamba, Embenx supports storing the initial hidden states ($h_0$) in a specialized ``StateCollection``.

.. code-block:: python

   from embenx.core import StateCollection
   
   col = StateCollection(name="mamba_ltm", dimension=128)
   
   # Add embeddings and their associated hidden states
   col.add_states(vectors, states=h_tensors)
   
   # Retrieve and prime
   results = col.search(query_vector)
   h0 = col.get_state(results[0][0])

ClusterKV Optimization
---------------------

For high-throughput scenarios, Embenx implements semantic clustering (as described in ClusterKV, arXiv:2412.03213) to optimize retrieval.

.. code-block:: python

   from embenx.core import ClusterCollection
   
   col = ClusterCollection(n_clusters=10, dimension=768)
   col.add(vectors, metadata)
   
   # Perform clustering
   col.cluster_data()
   
   # Optimized search
   results = col.search_clustered(query_vector, top_k=5)

Spatial Cognitive Memory (ESWM)
------------------------------

Inspired by neuroscience (ICLR 2026), Embenx supports spatial-aware episodic memory for agent navigation.

.. code-block:: python

   from embenx.core import SpatialCollection
   
   col = SpatialCollection(dimension=768)
   
   # Add semantic embeddings with [x, y, z] coordinates
   col.add_spatial(vectors, coords=xyz_positions)
   
   # Search for similar memories near the agent's current position
   results = col.search_spatial(query_vector, current_coords=my_pos, spatial_radius=5.0)

Temporal Episodic Memory (Echo)
------------------------------

Embenx supports time-aware retrieval (as described in Echo, arXiv:2502.16090), allowing for recency-biased search and time-window filtering.

.. code-block:: python

   from embenx.core import TemporalCollection
   import time
   
   col = TemporalCollection(dimension=768)
   
   # Add embeddings with timestamps (Unix epochs)
   col.add_temporal(vectors, timestamps=[time.time() - 3600, time.time()])
   
   # Search with recency bias (0.7 weight to recency)
   results = col.search_temporal(query_vector, recency_weight=0.7)
   
   # Search within a specific time window
   window = (start_time, end_time)
   results = col.search_temporal(query_vector, time_window=window)

Visual Explorer
--------------

Embenx includes a web-based UI to visualize your collections. It uses PCA or t-SNE to reduce high-dimensional vectors to 2D or 3D clusters.

.. code-block:: bash

   embenx explorer

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
