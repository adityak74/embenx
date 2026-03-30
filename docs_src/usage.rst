Usage Guide
===========

Embenx is designed to be simple yet powerful. This guide covers the most common usage patterns.

Core API
--------

The primary interface is the ``Collection`` class.

.. code-block:: python

   from embenx import Collection
   import numpy as np

   # 1. Initialize
   col = Collection(dimension=768, indexer_type="faiss-hnsw")

   # 2. Add data
   vectors = np.random.rand(100, 768).astype('float32')
   metadata = [{"id": i, "text": f"Doc {i}"} for i in range(100)]
   col.add(vectors, metadata)

   # 3. Search
   query_vector = np.random.rand(768).astype('float32')
   results = col.search(query_vector, top_k=5)

   # 4. Filtered Search
   results = col.search(query_vector, top_k=5, where={"id": 10})

   # 5. Serialization
   col.to_parquet("my_data.parquet")
   col2 = Collection.from_parquet("my_data.parquet")

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

Agentic Memory & Self-Healing
----------------------------

For autonomous agents, Embenx provides an ``AgenticCollection`` that supports feedback loops to automatically improve retrieval accuracy over time.

.. code-block:: python

   from embenx.core import AgenticCollection
   
   col = AgenticCollection(dimension=768)
   
   # Perform an agentic search (incorporates previous feedback)
   results = col.agentic_search(query_vector, top_k=5)
   
   # Provide feedback on a result
   # This will boost 'doc_123' in future searches for similar queries
   col.feedback(doc_id="doc_123", label="good")
   
   # Demote a noise result
   col.feedback(doc_id="doc_noise", label="bad")

Retrieval Zoo
-------------

Embenx provides a "Zoo" of pre-indexed collections for common research datasets.

.. code-block:: python

   from embenx.data import load_from_zoo
   
   # Automatically download and load SQuAD v2
   col = load_from_zoo("squad-v2")
   
   # Search immediately
   results = col.search(query_vector)

Visual Explorer
--------------

Embenx includes a web-based UI to visualize your collections. It uses PCA or t-SNE to reduce high-dimensional vectors to 2D or 3D clusters.

.. code-block:: bash

   embenx explorer

Hybrid Search (Dense + Sparse)
----------------------------

Embenx supports combining dense vector search with sparse BM25 retrieval.

.. code-block:: python

   # 1. Initialize with sparse indexer
   col = Collection(dimension=768, sparse_indexer_type="bm25")

   # 2. Add text and vectors
   col.add(vectors, metadata=[{"text": "Sample content"}])

   # 3. Hybrid search (uses Reciprocal Rank Fusion)
   results = col.hybrid_search(
       query_vector=q_vec,
       query_text="What is the content?",
       top_k=5,
       dense_weight=0.5,
       sparse_weight=0.5
   )
