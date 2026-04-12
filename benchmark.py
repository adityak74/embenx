import importlib.util
import inspect
import os
import time
from typing import Any, Dict, List

import psutil
from rich.console import Console
from rich.table import Table

from data import load_documents
from indexers import BaseIndexer, get_indexer_map
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
    subset: str = "default",  # Added as optional
):
    """
    Run Embenx benchmarks. Matches original signature for test compatibility.
    """
    # Load Data
    console.print(f"\n[bold]Loading up to {max_docs} documents from {dataset_name}...[/bold]")

    # Check if dataset_name is actually a path (Parquet benchmark use case)
    if os.path.exists(dataset_name) and dataset_name.endswith(".parquet"):
        from core import Collection

        col = Collection.from_parquet(dataset_name)
        docs = col._metadata
        embeddings = col._vectors.tolist()
        dimension = col.dimension
    else:
        # Standard HF/Zoo load
        docs = load_documents(dataset_name, subset, split, max_docs)

        if not docs:
            console.print("[red]No documents loaded. Exiting.[/red]")
            return
        console.print(f"Loaded {len(docs)} documents.")

        # Embed Data
        console.print(f"\n[bold]Generating embeddings using LiteLLM ({model_name})...[/bold]")
        embedder = Embedder(model_name)

        text_field = text_column
        if text_field not in docs[0] and "text" in docs[0]:
            text_field = "text"
        elif text_field not in docs[0] and "content" in docs[0]:
            text_field = "content"

        texts = [d.get(text_field, str(d)) for d in docs]

        t0 = time.perf_counter()
        embeddings = embedder.embed_texts(texts)
        emb_time = time.perf_counter() - t0

        if not embeddings:
            console.print("[red]Failed to generate embeddings.[/red]")
            return

        dimension = len(embeddings[0])
        console.print(
            f"Generated {len(embeddings)} embeddings of dimension {dimension} in {emb_time:.2f}s."
        )

    # Initialize Indexers
    indexers_map = get_indexer_map()

    if custom_indexer_script:
        custom_name, custom_cls = load_custom_indexer(custom_indexer_script, console)
        if custom_cls:
            c_name_lower = custom_name.lower()
            indexers_map[c_name_lower] = custom_cls
            console.print(
                f"[green]✓[/green] Successfully loaded custom indexer: [bold]{custom_name}[/bold]"
            )
            if c_name_lower not in [x.lower() for x in indexer_names]:
                indexer_names.append(c_name_lower)

    results = []
    for name in indexer_names:
        name_lower = name.lower()
        if name_lower not in indexers_map:
            console.print(f"[yellow]Warning: Indexer '{name}' not found. Skipping.[/yellow]")
            continue

        res = benchmark_single_indexer(
            name, indexers_map[name_lower], dimension, embeddings, docs, console, cleanup
        )
        if res:
            results.append(res)

    # Report
    if results:
        display_results(results, console)
        return results
    return []


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


def generate_report(
    results: List[Dict[str, Any]], dataset_name: str, output_path: str = "benchmark_report.md"
):
    """
    Generate a formatted Markdown technical report from benchmark results.
    """
    import datetime

    report = []
    report.append("# Embenx Retrieval Benchmark Report 🚀")
    report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: **{dataset_name}**")
    report.append("\n## Executive Summary")

    if not results:
        report.append("No results to report.")
    else:
        # Find winners
        query_times = [float(r["Query Time (ms)"]) for r in results]
        fastest_idx = query_times.index(min(query_times))
        fastest = results[fastest_idx]["Indexer"]

        sizes = [float(r["Index Size (KB)"]) for r in results]
        smallest_idx = sizes.index(min(sizes))
        smallest = results[smallest_idx]["Indexer"]

        report.append(f"- **Fastest Indexer**: {fastest} ({min(query_times):.2f} ms/query)")
        report.append(f"- **Most Memory Efficient**: {smallest} ({min(sizes):.2f} KB)")

        report.append("\n## Results Table")
        report.append(
            "| Indexer | Build Time (s) | Query Time (ms) | Index Size (KB) | Memory Diff (MB) |"
        )
        report.append("| :--- | :--- | :--- | :--- | :--- |")

        for r in results:
            report.append(
                f"| {r['Indexer']} | {r['Build Time (s)']} | {r['Query Time (ms)']} | {r['Index Size (KB)']} | {r['Memory Diff (MB)']} |"
            )

        report.append("\n## Analysis & Recommendations")
        report.append("Based on the data above, we recommend:")
        if "FAISS-HNSW" in [r["Indexer"] for r in results]:
            report.append(
                "- Use **FAISS-HNSW** for production-grade local search balancing speed and memory."
            )
        if "SCANN" in [r["Indexer"] for r in results]:
            report.append(
                "- Use **ScaNN** for state-of-the-art speed/recall if on supported hardware."
            )
        report.append(
            "- For ultra-low latency requirements, prioritize indexers with sub-1ms query times."
        )

    with open(output_path, "w") as f:
        f.write("\n".join(report))

    return output_path
