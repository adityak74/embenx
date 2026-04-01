CLI Reference
=============

Embenx provides a powerful Command Line Interface (CLI) for benchmarking, setup, and ecosystem interaction.

General Commands
----------------

info
~~~~
Display system information and status of installed vector backends.

.. code-block:: bash

   embenx info

check
~~~~~
Verify environment health and dependency availability.

.. code-block:: bash

   embenx check

setup
~~~~~
Setup the environment and pull specific models (for Ollama).

.. code-block:: bash

   embenx setup --model ollama/nomic-embed-text --pull

Benchmarking
------------

benchmark
~~~~~~~~~
Run performance benchmarks on local Parquet files.

.. code-block:: bash

   embenx benchmark data.parquet --indexers faiss,usearch --max-docs 1000

grand-benchmark
~~~~~~~~~~~~~~~
Run comprehensive benchmarks across all datasets in the Retrieval Zoo and generate a Grand Technical Report.

.. code-block:: bash

   embenx grand-benchmark --indexers faiss,scann,hnswlib

Agentic & Visual
----------------

mcp-start
~~~~~~~~~
Start the Model Context Protocol (MCP) server to connect AI agents.

.. code-block:: bash

   embenx mcp-start

explorer
~~~~~~~~
Launch the Streamlit web UI for data visualization and HNSW graph inspection.

.. code-block:: bash

   embenx explorer

Retrieval Zoo
-------------

zoo-list
~~~~~~~~
List available pre-indexed collections in the cloud zoo.

.. code-block:: bash

   embenx zoo-list

zoo-load
~~~~~~~~
Download and load a pre-built collection into local cache.

.. code-block:: bash

   embenx zoo-load squad-v2

Maintenance
-----------

cleanup
~~~~~~~
Remove temporary benchmark artifacts and database files to free up space.

.. code-block:: bash

   embenx cleanup

init-skill
~~~~~~~~~~
Initialize the Embenx skill for Gemini CLI sub-agents.

.. code-block:: bash

   embenx init-skill
