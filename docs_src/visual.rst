Visual Tools & Dashboard
=====================

Embenx provides a suite of interactive tools to help you understand your embeddings, visualize index structures, and test retrieval quality in real-time.

Launching the Explorer
---------------------

The primary dashboard is the **Embenx Explorer**, a Streamlit-based web UI. You can launch it directly from the CLI:

.. code-block:: bash

   embenx explorer

The dashboard will automatically detect any `.parquet` collections in your current directory.

Vector Clusters
--------------

The **Vector Clusters** tab provides a high-level view of your data distribution.

*   **Dimensionality Reduction**: Choose between **PCA** (fast, linear) or **t-SNE** (better for complex clusters) to project your vectors into 2D or 3D space.
*   **Metadata Inspection**: Hover over any point in the scatter plot to see its associated metadata fields.
*   **Cluster Identification**: Visually identify semantic groups and outliers in your dataset.

HNSW Visualizer 🕸️
-----------------

The **HNSW Visualizer** tab offers a unique look at the internal architecture of Hierarchical Navigable Small World graphs.

*   **Layer Visualization**: See how nodes are distributed across different layers of the hierarchy.
*   **Navigation Paths**: Visualize the "hops" the search algorithm takes through the graph to reach the nearest neighbors.
*   **Structure Audit**: Understand the connectivity and density of your index, which is critical for tuning parameters like ``M`` and ``efConstruction``.

RAG Playground 💬
----------------

The **RAG Playground** allows you to test Retrieval-Augmented Generation (RAG) loops without writing any code.

1.  **Select a Collection**: Choose the data you want to query.
2.  **Configure the LLM**: Enter an LLM model string (compatible with LiteLLM, e.g., ``ollama/llama3`` or ``gpt-4o``).
3.  **Chat**: Ask a question. Embenx will:
    *   Embed your query.
    *   Retrieve the top-K relevant documents from your collection.
    *   Inject the context into a prompt.
    *   Generate a response using the selected LLM.
4.  **Inspect Context**: Expand the "Show Retrieved Context" section to see exactly what information was provided to the model, helping you debug retrieval precision issues.

Interactive Search
-----------------

The **Interactive Search** tab is a dedicated area for raw retrieval testing.

*   **Real-time Results**: Type a query and instantly see the metadata and distance scores for the top results.
*   **Score Tuning**: Adjust ``top_k`` on the fly to see how it affects the result set.
*   **Filtering**: (Upcoming) Apply metadata filters directly in the UI to test subset retrieval.

One-Click Dashboarding
---------------------

Because the Explorer runs on Streamlit, it is easy to deploy as a permanent dashboard for your team. Simply host the ``explorer.py`` file on any Python-capable server or container.
