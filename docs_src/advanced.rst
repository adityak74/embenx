Advanced Research & Agentic Memory
==================================

Embenx implements several state-of-the-art algorithms and specialized memory structures to handle the complex requirements of autonomous agents and high-throughput retrieval.

Research-Driven Optimizations
----------------------------

Embenx stays at the cutting edge by integrating recently published research directly into the toolkit.

ClusterKV (arXiv:2412.03213)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ClusterKV** improves retrieval throughput by semantic grouping. Instead of searching the entire index, Embenx can identify the most relevant semantic "neighborhood" and focus its compute there.

.. code-block:: python

   from embenx.core import ClusterCollection
   
   col = ClusterCollection(n_clusters=10, dimension=768)
   col.add(vectors, metadata)
   col.cluster_data()  # Computes K-Means centroids
   
   # Searches within the nearest cluster for speed
   results = col.search_clustered(query_vector)

TurboQuant (arXiv:2504.19874)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**TurboQuant** addresses the "Memory Wall" in LLM serving by aggressively compressing activation tensors (KV cache). Embenx uses a sign-based 1-bit quantization scheme to reduce activation storage by up to 4x while maintaining high signal.

.. code-block:: python

   # Activations are automatically packed into int8 safe-tensors
   col.add_cache(vectors, activations, metadata, quantize=True)

Agentic Memory structures
------------------------

Autonomous agents require memory that is self-healing, time-aware, and spatially grounded.

Self-Healing Retrieval
~~~~~~~~~~~~~~~~~~~~~~

The ``AgenticCollection`` allows agents to provide feedback on retrieval quality, which is then used to bias future results.

*   **Positive Feedback**: Decreases the semantic distance of a document for future queries.
*   **Negative Feedback**: Increases the distance, effectively demoting "noise" or irrelevant results.

.. code-block:: python

   col = AgenticCollection(dimension=768)
   col.feedback(doc_id="doc_abc", label="good")
   results = col.agentic_search(query_vector)

Temporal Memory (Echo, arXiv:2502.16090)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agents often need to remember "what happened when." The ``TemporalCollection`` supports recency-biased retrieval and time-window filtering.

.. code-block:: python

   col = TemporalCollection(dimension=768)
   col.add_temporal(vectors, timestamps=my_unix_timestamps)
   
   # recency_weight=0.7 gives high priority to recent events
   results = col.search_temporal(query_vector, recency_weight=0.7)

Spatial Memory (ESWM, ICLR 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspired by neuroscience, **Episodic Spatial World Memory (ESWM)** grounds embeddings in physical coordinates. This is essential for embodied agents (robots, drones) that need to retrieve memories based on their current location.

.. code-block:: python

   col = SpatialCollection(dimension=768)
   col.add_spatial(vectors, coords=xyz_positions)
   
   # Find memories near the agent
   results = col.search_spatial(query_vector, current_coords=[0, 0, 0], spatial_radius=10.0)

Managed Sessions
~~~~~~~~~~~~~~~~

The ``Session`` class provides a high-level manager for agentic memory, handling persistence and temporal decay automatically. Each session is stored as a dedicated Parquet file, making multi-user agent systems easy to scale.

.. code-block:: python

   from embenx.core import Session
   
   sess = Session(session_id="agent_alpha", dimension=768)
   sess.add_interaction(vector, "Agent thought or observation")
   context = sess.retrieve_context(query_vec)
