import os
import time
import importlib.util
import inspect
from typing import List

import psutil
from rich.console import Console
from rich.table import Table

from data import load_documents
from indexers.base import BaseIndexer
from indexers.chroma_indexer import ChromaIndexer
from indexers.duckdb_indexer import DuckDBIndexer
from indexers.faiss_indexer import FaissIndexer
from indexers.lance_indexer import LanceIndexer
from indexers.milvus_indexer import MilvusIndexer
from indexers.qdrant_indexer import QdrantIndexer
from indexers.weaviate_indexer import WeaviateIndexer
from llm import Embedder


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def load_custom_indexer(script_path: str, console: Console):
    """
    Dynamically load a class inheriting from BaseIndexer from a given script.
    """
    try:
        spec = importlib.util.spec_from_file_location("custom_indexer", script_path)
        if spec is None or spec.loader is None:
            console.print(f"[red]Could not load spec for {script_path}[/red]")
            return None, None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseIndexer) and obj is not BaseIndexer:
                return name, obj
                
        console.print(f"[red]No class inheriting from BaseIndexer found in {script_path}[/red]")
        return None, None
    except Exception as e:
        console.print(f"[red]Error loading custom indexer from {script_path}: {e}[/red]")
        return None, None


def benchmark_single_indexer(
    name, indexer_cls, dimension, embeddings, metadata, console, cleanup=True
):
    console.print(f"\n[bold cyan]--- Benchmarking {name.upper()} ---[/bold cyan]")
    indexer = indexer_cls(dimension=dimension)

    # Build Index
    mem_before = get_memory_usage()
    t0 = time.perf_counter()
    try:
        indexer.build_index(embeddings, metadata)
        build_time = time.perf_counter() - t0
    except Exception as e:
        console.print(f"[red]Failed to build index for {name}: {e}[/red]")
        return None

    mem_after = get_memory_usage()
    mem_diff = mem_after - mem_before
    index_size = indexer.get_size()

    # Query Benchmarking
    query_embeddings = embeddings[: min(10, len(embeddings))]
    query_time = 0
    if query_embeddings:
        t0 = time.perf_counter()
        for q_emb in query_embeddings:
            indexer.search(q_emb, top_k=5)
        query_time = (time.perf_counter() - t0) / len(query_embeddings) * 1000  # ms per query

    result = {
        "Indexer": name.upper(),
        "Build Time (s)": f"{build_time:.4f}",
        "Query Time (ms)": f"{query_time:.2f}",
        "Index Size (KB)": f"{index_size / 1024:.2f}",
        "Memory Diff (MB)": f"{mem_diff:.2f}",
    }

    if cleanup:
        indexer.cleanup()

    console.print(f"Done {name.upper()}.")
    return result


def run_benchmark(
    dataset_name: str,
    split: str,
    text_column: str,
    max_docs: int,
    indexer_names: List[str],
    model_name: str,
    console: Console,
    data_files: str = None,
    cleanup: bool = True,
    custom_indexer_script: str = None,
):
    # Load Data
    console.print(f"\n[bold]Loading up to {max_docs} documents...[/bold]")
    docs = load_documents(dataset_name, split, text_column, max_docs, data_files=data_files)

    if not docs:
        console.print("[red]No documents loaded. Exiting.[/red]")
        return
    console.print(f"Loaded {len(docs)} documents.")

    # Embed Data
    console.print(f"\n[bold]Generating embeddings using LiteLLM ({model_name})...[/bold]")
    embedder = Embedder(model_name)
    texts = [d["text"] for d in docs]
    metadata = [d["metadata"] for d in docs]

    t0 = time.perf_counter()
    embeddings = embedder.embed_texts(texts)
    emb_time = time.perf_counter() - t0

    if not embeddings:
        console.print("[red]Failed to generate embeddings. Is Ollama running?[/red]")
        return

    dimension = len(embeddings[0])
    console.print(
        f"Generated {len(embeddings)} embeddings of dimension {dimension} in {emb_time:.2f}s."
    )
    console.print(f"Approximate tokens processed: {embedder.total_tokens_approx}")

    # Initialize Indexers
    indexers_map = {
        "faiss": FaissIndexer,
        "chroma": ChromaIndexer,
        "qdrant": QdrantIndexer,
        "milvus": MilvusIndexer,
        "lance": LanceIndexer,
        "weaviate": WeaviateIndexer,
        "duckdb": DuckDBIndexer,
    }

    if custom_indexer_script:
        custom_name, custom_cls = load_custom_indexer(custom_indexer_script, console)
        if custom_cls:
            c_name_lower = custom_name.lower()
            indexers_map[c_name_lower] = custom_cls
            console.print(f"[green]✓[/green] Successfully loaded custom indexer: [bold]{custom_name}[/bold]")
            if c_name_lower not in [x.lower() for x in indexer_names] and "all" not in [x.lower() for x in indexer_names]:
                indexer_names.append(c_name_lower)

    results = []
    for name in indexer_names:
        name_lower = name.lower()
        if name_lower not in indexers_map:
            console.print(f"[yellow]Warning: Indexer '{name}' not found. Skipping.[/yellow]")
            continue

        res = benchmark_single_indexer(
            name, indexers_map[name_lower], dimension, embeddings, metadata, console, cleanup
        )
        if res:
            results.append(res)

    # Report
    if results:
        display_results(results, console)


def display_results(results, console):
    console.print("\n[bold green]Benchmark Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Indexer", style="cyan")
    table.add_column("Build Time (s)", justify="right")
    table.add_column("Query Time (ms/query)", justify="right")
    table.add_column("Index Size (KB)", justify="right")
    table.add_column("Memory Added (MB)", justify="right")

    for r in results:
        table.add_row(
            r["Indexer"],
            r["Build Time (s)"],
            r["Query Time (ms)"],
            r["Index Size (KB)"],
            r["Memory Diff (MB)"],
        )
    console.print(table)
