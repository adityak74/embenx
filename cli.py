import os
import subprocess
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mcp_server import run

app = typer.Typer(help="Embenx: Universal Embedding Retrieval Toolkit & Benchmark.")
console = Console()


@app.command()
def info():
    """
    Display Embenx system information and installed backends.
    """
    console.print(
        Panel.reveal(
            "[bold rocket] Embenx v1.5.1[/bold rocket]\n[dim]The Agentic Memory Layer[/dim]",
            title="System Info",
            expand=False,
        )
    )

    from indexers import get_indexer_map

    indexer_map = get_indexer_map()

    table = Table(title="Available Indexers", show_header=True, header_style="bold magenta")
    table.add_column("Indexer", style="cyan")
    table.add_column("Status", justify="center")

    for name in indexer_map.keys():
        try:
            if name == "faiss":
                import faiss
            elif name == "usearch":
                import usearch
            elif name == "hnswlib":
                import hnswlib
            elif name == "annoy":
                import annoy
            elif name == "scann":
                import scann
            elif name == "chroma":
                import chromadb
            elif name == "qdrant":
                import qdrant_client
            elif name == "milvus":
                import pymilvus
            elif name == "lance":
                import lancedb
            elif name == "duckdb":
                import duckdb
            elif name == "weaviate":
                import weaviate
            elif name == "pgvector":
                import psycopg2
            elif name == "opensearch":
                import opensearchpy
            elif name == "elasticsearch":
                import elasticsearch
            elif name == "vespa":
                import requests  # Vespa uses requests
            elif name == "bm25":
                import rank_bm25

            table.add_row(name, "[green]✓ Ready[/green]")
        except ImportError:
            table.add_row(name, "[red]✖ Missing[/red]")

    console.print(table)


@app.command()
def setup(
    model: str = typer.Option("ollama/nomic-embed-text", help="Embedding model to use."),
    pull: bool = typer.Option(False, help="Whether to pull the model if using Ollama."),
):
    """
    Setup the environment and verify model availability.
    """
    console.print("Embenx Environment Check")
    console.print(f"[yellow]Setting up Embenx with model: {model}...[/yellow]")

    if model.startswith("ollama/"):
        model_name = model.replace("ollama/", "")
        try:
            if pull:
                console.print(f"[cyan]Pulling model {model_name} via Ollama...[/cyan]")
                subprocess.run(["ollama", "pull", model_name], check=True)

            # Verify model is available
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model_name in result.stdout:
                console.print(f"[bold green]✓ Model '{model_name}' is available[/bold green]")
            else:
                console.print("[bold red]✖ Model not found in Ollama.[/bold red]")
                console.print("[dim]Run with --pull to download it automatically.[/dim]")
        except Exception as e:
            console.print(f"[bold red]Ollama error: {e}[/bold red]")

    console.print("\n[bold green]✓ Setup complete.[/bold green]")


@app.command()
def benchmark(
    path: Optional[str] = typer.Argument(None, help="Path to local Parquet file."),
    dataset: str = typer.Option("dummy", "--dataset", "-d", help="Hugging Face dataset name."),
    subset: str = typer.Option("default", "--subset", "-s", help="Dataset subset."),
    split: str = typer.Option("train", help="Dataset split."),
    text_col: str = typer.Option("text", "--text-col", "-t", help="Column containing text."),
    vector_col: str = typer.Option("vector", "--vector-col", help="Column containing vectors."),
    max_docs: int = typer.Option(100, "--max-docs", "-n", help="Maximum documents to index."),
    indexers: str = typer.Option(
        "faiss,simple", "--indexers", "-i", help="Comma-separated list of indexers."
    ),
    top_k: int = typer.Option(5, help="Number of neighbors to search."),
    custom_indexer: Optional[str] = typer.Option(
        None, "--custom-indexer", help="Path to custom indexer script."
    ),
    report: bool = typer.Option(False, help="Generate a Markdown report."),
    model: str = typer.Option("ollama/nomic-embed-text", help="Embedding model."),
):
    """
    Run Embenx benchmarks on local or remote data.
    """
    from benchmark import generate_report, run_benchmark

    indexer_list = indexers.split(",") if indexers != "all" else None

    console.print("Run Embenx benchmarks")
    console.print("[bold green]Starting Embenx Benchmark...[/bold green]")

    # Matching original signature: run_benchmark(dataset_name, split, text_column, max_docs, indexer_names, model_name, console, ...)
    results = run_benchmark(
        dataset if not path else path,
        split,
        text_col,
        max_docs,
        indexer_list,
        model,
        console,
        custom_indexer_script=custom_indexer,
        subset=subset,
    )

    if report and results:
        path = generate_report(results, dataset if not path else path)
        console.print(f"[bold green]✓ Report generated: {path}[/bold green]")


@app.command()
def grand_benchmark(
    indexers: str = typer.Option("faiss,simple", "--indexers", "-i", help="Indexers to test."),
    max_docs: int = typer.Option(100, "--max-docs", "-n", help="Docs per dataset."),
    model: str = typer.Option("ollama/nomic-embed-text", help="Embedding model."),
):
    """
    Run benchmarks across all Retrieval Zoo datasets and generate a Grand Report.
    """
    from benchmark import generate_report, run_benchmark
    from data import list_zoo

    datasets = list_zoo()
    all_results = []

    indexer_list = indexers.split(",") if indexers != "all" else None

    for ds in datasets:
        console.print(f"\n[bold magenta]>>> Benchmarking Zoo Dataset: {ds}[/bold magenta]")
        res = run_benchmark(ds, "train", "text", max_docs, indexer_list, model, console)
        if res:
            for r in res:
                r["Dataset"] = ds
            all_results.extend(res)

    if all_results:
        path = generate_report(all_results, " Retrieval Zoo (Grand)")
        console.print(f"\n[bold green]✓ Grand Technical Report generated: {path}[/bold green]")


@app.command()
def cleanup():
    """
    Remove temporary benchmark artifacts and database files.
    """
    import glob

    console.print("[yellow]Cleaning up Embenx artifacts...[/yellow]")
    patterns = ["*.db", "*.index", "*.bin", ".embenx_cache/"]
    found_any = False
    for pattern in patterns:
        files = glob.glob(pattern)
        for f in files:
            found_any = True
            try:
                if os.path.isdir(f):
                    import shutil

                    shutil.rmtree(f)
                else:
                    os.remove(f)
                console.print(f" - Successfully removed {f}")
            except Exception as e:
                console.print(f" [red]Failed to remove {f}: {e}[/red]")

    if not found_any:
        console.print("No artifacts found.")

    console.print("[bold green]✓ Cleanup complete.[/bold green]")


@app.command()
def init_skill():
    """
    Initialize the Embenx skill for Gemini CLI sub-agents.
    """
    skill_content = """# Embenx Skill 🚀
Optimized for high-performance embedding retrieval and benchmarking.
Use `Collection` for high-level operations.
    """
    with open("SKILL.md", "w") as f:
        f.write(skill_content)
    console.print("[bold green]Created SKILL.md successfully[/bold green]")


@app.command()
def mcp_start():
    """
    Start the Embenx MCP server for agentic tool-use.
    """
    import asyncio

    console.print("[bold green]Starting Embenx MCP Server...[/bold green]")
    console.print("[cyan]Connect your agent (Claude Desktop, etc.) via stdio.[/cyan]")
    asyncio.run(run())


@app.command()
def explorer():
    """
    Launch the Embenx Explorer web UI to visualize collections.
    """
    import subprocess

    console.print("[bold green]Launching Embenx Explorer...[/bold green]")
    try:
        subprocess.run(["streamlit", "run", "explorer.py"], check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Explorer stopped.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error launching explorer: {e}[/bold red]")


@app.command()
def zoo_list():
    """
    List all available pre-indexed collections in the Embenx Retrieval Zoo.
    """
    from data import list_zoo

    datasets = list_zoo()
    console.print("[bold cyan]Embenx Retrieval Zoo[/bold cyan]")
    for ds in datasets:
        console.print(f" - {ds}")


@app.command()
def zoo_load(name: str):
    """
    Download and load a collection from the Embenx Retrieval Zoo.
    """
    from data import load_from_zoo

    try:
        console.print(f"[yellow]Loading {name} from zoo...[/yellow]")
        col = load_from_zoo(name)
        console.print(f"[bold green]✓ Successfully loaded {name}[/bold green]")
        console.print(f"Size: {len(col._metadata)} documents")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def list_indexers():
    """
    List available indexing libraries for benchmarking.
    """
    from indexers import get_indexer_map

    indexer_map = get_indexer_map()
    console.print("[bold cyan]Available Indexers:[/bold cyan]")
    for name in indexer_map.keys():
        console.print(f" - {name}")


@app.command()
def check():
    """
    Verify environment and dependencies.
    """
    from indexers import get_indexer_map

    indexer_map = get_indexer_map()
    all_ok = True

    console.print("[bold cyan]Dependency Check:[/bold cyan]")
    for name in indexer_map.keys():
        try:
            if name == "faiss":
                import faiss
            elif name == "usearch":
                import usearch
            console.print(f" - {name}: [green]Installed[/green]")
        except ImportError:
            console.print(f" - {name}: [yellow]Not found[/yellow]")
            all_ok = False

    console.print(f"\n{'[bold green]✓ Ready!' if all_ok else '[bold yellow]⚠ Fix issues above.'}")


if __name__ == "__main__":
    app()
