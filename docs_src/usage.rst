Usage Guide
===========

Embenx provides a simple CLI for benchmarking.

Benchmark Command
-----------------

The primary command is ``benchmark``:

.. code-block:: bash

   embenx benchmark --dataset <dataset_name> [options]

Options:

* ``--dataset`` / ``-d``: HuggingFace dataset name or format (csv, json).
* ``--max-docs`` / ``-m``: Maximum documents to index.
* ``--indexers`` / ``-i``: Comma-separated list of indexers to test.
* ``--model``: LiteLLM model string (e.g., ``ollama/nomic-embed-text``).
* ``--custom-indexer``: Path to a custom indexer Python script.

Environment Setup
-----------------

Check your environment before running:

.. code-block:: bash

   embenx setup --pull

Custom Indexers
---------------

You can create a custom indexer by inheriting from ``BaseIndexer``:

.. code-block:: python

   from indexers import BaseIndexer

   class MyIndexer(BaseIndexer):
       def build_index(self, embeddings, metadata):
           # build logic
           pass

       def search(self, query_embedding, top_k=5):
           # search logic
           return []

       def get_size(self):
           return 0
