Usage Guide
===========

Embenx is designed to be simple for prototyping yet robust enough for research-grade agentic memory. This guide covers all primary features.

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
   vectors = np.random.rand(100, 768).astype('float32')
   metadata = [{"id": i, "text": f"Document {i}", "tag": "test"} for i in range(100)]
   col.add(vectors, metadata)

   # 3. Basic Search
   results = col.search(query_vector, top_k=5)

   # 4. Metadata Filtering
   # Supports exact match dictionary filters
   results = col.search(query_vector, top_k=5, where={"tag": "test"})

   # 5. Serialization
   # Saves to a portable Parquet file containing both vectors and metadata
   col.to_parquet("my_memory.parquet")
   
   # Load back
   new_col = Collection.from_parquet("my_memory.parquet")

Advanced Retrieval
------------------

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
   
   # Metadata MUST contain a 'text' field for BM25
   col.add(vectors, metadata=[{"text": "The quick brown fox", "id": 1}])
   
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

Agentic Memory
--------------

Model Context Protocol (MCP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Embenx can act as a native tool for AI agents.

.. code-block:: bash

   # Start the MCP server via CLI
   embenx mcp-start

Configure your agent (e.g., Claude Desktop) to point to this server to give it long-term memory capabilities.

Self-Healing Retrieval
~~~~~~~~~~~~~~~~~~~~~~

Use ``AgenticCollection`` to implement feedback loops that improve ranking based on user or agent feedback.

.. code-block:: python

   from embenx.core import AgenticCollection
   
   col = AgenticCollection(dimension=768)
   
   # Search incorporating previous feedback
   results = col.agentic_search(query_vector)
   
   # Provide feedback to boost a specific document
   col.feedback(doc_id="doc_123", label="good")
   
   # Provide negative feedback to demote noise
   col.feedback(doc_id="noise_456", label="bad")

Managed Sessions
~~~~~~~~~~~~~~~~

Use the ``Session`` class for conversational memory that automatically persists and handles temporal decay.

.. code-block:: python

   from embenx.core import Session
   
   sess = Session(session_id="user_001", dimension=768)
   
   # Log interactions
   sess.add_interaction(vector, "What is the weather today?", role="user")
   
   # Retrieve context with recency bias
   context = sess.retrieve_context(new_query_vec, recency_weight=0.6)

Research-Driven Optimizations
-----------------------------

ClusterKV Optimization
~~~~~~~~~~~~~~~~~~~~~~

For high-throughput applications, group your vectors into semantic clusters to accelerate retrieval.

.. code-block:: python

   from embenx.core import ClusterCollection
   
   col = ClusterCollection(n_clusters=10, dimension=768)
   col.add(vectors, metadata)
   
   # Build clusters
   col.cluster_data()
   
   # Search within the most relevant cluster
   results = col.search_clustered(query_vector)

TurboQuant (1-bit Quantization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggressively compress KV cache activation tensors using sign-based quantization.

.. code-block:: python

   from embenx.core import CacheCollection
   
   col = CacheCollection(name="my_cache", dimension=768)
   
   # Store large activation tensors with 4x compression
   col.add_cache(vectors, activations={"k": k_tensors, "v": v_tensors}, quantize=True)

World Models
------------

Trajectory Retrieval
~~~~~~~~~~~~~~~~~~~~

Search for sequences of vectors (state/action trajectories) using mean or max pooling.

.. code-block:: python

   # Define a sequence of vectors
   trajectory = np.array([[0.1, ...], [0.2, ...]])
   
   # Search for the closest matching experience
   results = col.search_trajectory(trajectory, pooling="mean")

Spatial Cognitive Memory (ESWM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Store memories with spatial coordinates and retrieve them based on distance to the agent.

.. code-block:: python

   from embenx.core import SpatialCollection
   
   col = SpatialCollection(dimension=768)
   col.add_spatial(vectors, coords=xyz_array)
   
   # Search near the current agent position
   results = col.search_spatial(query_vector, current_coords=[1.0, 5.0, 0.0])

Multimodal Retrieval
--------------------

Index and search images alongside text using CLIP-style models.

.. code-block:: python

   col = Collection(dimension=512)
   
   # Add local images
   col.add_images(["path/to/cat.jpg", "path/to/dog.png"])
   
   # Search using an image query
   results = col.search_image("query_image.jpg")

Visual & DevTools
-----------------

Embenx Explorer
~~~~~~~~~~~~~~~

Launch the built-in web UI to inspect your data.

.. code-block:: bash

   embenx explorer

The Explorer includes:
* **Vector Clusters**: PCA/t-SNE 3D visualization.
* **HNSW Visualizer**: Interactive graph of navigation layers.
* **RAG Playground**: Chat loop to test retrieval quality with LLMs.

Production Export
~~~~~~~~~~~~~~~~~

Move from local prototype to cloud-scale with one command.

.. code-block:: python

   # Sync local data to a remote Qdrant instance
   col.export_to_production(backend="qdrant", connection_url="http://remote-qdrant:6333")
