import typer
from rich.console import Console

app = typer.Typer(help="Embenx 🚀: Benchmark vector indexing libraries with ease")
console = Console()


@app.command()
def benchmark(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="HuggingFace dataset name or data format (e.g., 'csv', 'json', 'parquet')",
    ),
    split: str = typer.Option("train", "--split", "-s", help="Dataset split to use"),
    text_column: str = typer.Option(
        "text", "--text-column", "-c", help="Column containing the text to index"
    ),
    max_docs: int = typer.Option(
        1000, "--max-docs", "-m", help="Maximum number of documents to index"
    ),
    indexers: str = typer.Option(
        "all",
        "--indexers",
        "-i",
        help="Comma-separated list or 'all'",
    ),
    model: str = typer.Option(
        "ollama/nomic-embed-text",
        "--model",
        help="LiteLLM model name (e.g., 'ollama/nomic-embed-text', 'openai/text-embedding-3-small')",
    ),
    data_files: str = typer.Option(
        None, "--data-files", help="Path to local data files (for CSV/JSON formats)"
    ),
    cleanup: bool = typer.Option(
        True,
        "--cleanup/--no-cleanup",
        help="Automatically cleanup temporary index files after benchmarking",
    ),
    custom_indexer: str = typer.Option(
        None, "--custom-indexer", help="Path to a Python script containing a custom indexer class"
    ),
):
    """
    Run Embenx benchmarks across different vector indexing libraries.
    """
    console.print(f"[bold green]Starting Embenx benchmark...[/bold green]")
    console.print(f"Dataset: [cyan]{dataset}[/cyan] ({split})")
    if data_files:
        console.print(f"Data Files: [cyan]{data_files}[/cyan]")
    if not cleanup:
        console.print(f"Cleanup: [yellow]Disabled[/yellow]")
    if custom_indexer:
        console.print(f"Custom Indexer: [cyan]{custom_indexer}[/cyan]")
    console.print(f"Max Docs: [cyan]{max_docs}[/cyan]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"Indexers: [cyan]{indexers}[/cyan]")

    # We will import the benchmark engine here to avoid early loading overhead
    from benchmark import run_benchmark
    from indexers import get_indexer_map

    indexers_map = get_indexer_map()
    all_available = list(indexers_map.keys())

    # Parse indexers
    if indexers.lower() == "all":
        selected_indexers = all_available
    else:
        selected_indexers = [x.strip().lower() for x in indexers.split(",")]

    run_benchmark(
        dataset_name=dataset,
        split=split,
        text_column=text_column,
        max_docs=max_docs,
        indexer_names=selected_indexers,
        model_name=model,
        console=console,
        data_files=data_files,
        cleanup=cleanup,
        custom_indexer_script=custom_indexer,
    )


@app.command()
def init_skill():
    """
    Generate a SKILL.md file for AI agents (Claude, Gemini, etc.) to use Embenx.
    """
    skill_content = """# Embenx Skill 🚀

This skill enables the agent to perform high-performance benchmarking of vector indexing libraries using the **Embenx** CLI.

## <instructions>
- **Objective**: Assist the user in evaluating and comparing vector search backends.
- **Core Workflow**:
    1. **Identify Requirements**: Determine the user's dataset and the indexing libraries they wish to compare.
    2. **Check Environment**: Always run `uv run embenx setup` first. Pass `--pull` to auto-pull a missing Ollama model.
    3. **Execute Benchmark**:
        - For standard HF datasets: `uv run embenx benchmark --dataset <name> --max-docs <num>`.
        - For local data: `uv run embenx benchmark --dataset json --data-files <path> --text-column <col>`.
    4. **Analyze Results**: Review the generated table and provide a technical recommendation.
    5. **Cleanup**: Run `uv run embenx cleanup` after benchmarking.
- **Safety**: Default to a small `--max-docs` (e.g., 100) for initial testing.
</instructions>

## <available_resources>
- **Documentation**: `docs/index.html`.
- **CLI Help**: `uv run embenx --help`.
- **Examples**: `examples/`.
</available_resources>
"""

    try:
        with open("SKILL.md", "w") as f:
            f.write(skill_content)
        console.print("[bold green]✓ Created SKILL.md successfully.[/bold green]")
        console.print("[cyan]AI agents can now activate this skill to use Embenx effectively.[/cyan]")
    except Exception as e:
        console.print(f"[bold red]✗ Failed to create SKILL.md: {e}[/bold red]")


@app.command()
def cleanup():
    """
    Manually remove any leftover benchmark artifacts (*.db, *.lance, etc.)
    """
    import glob
    import os
    import shutil

    console.print("[bold yellow]Cleaning up benchmark artifacts...[/bold yellow]")

    patterns = ["*.db", "*.lance", "benchmark", "benchmark.lance", "weaviate_data"]
    removed_count = 0

    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                console.print(f" [green]✓[/green] Removed: {path}")
                removed_count += 1
            except Exception as e:
                console.print(f" [red]✗[/red] Failed to remove {path}: {e}")

    if removed_count == 0:
        console.print("[cyan]No artifacts found. Workspace is clean.[/cyan]")
    else:
        console.print(f"[bold green]Successfully removed {removed_count} artifacts.[/bold green]")


@app.command()
def setup(
    model: str = typer.Option(
        "ollama/nomic-embed-text",
        "--model",
        help="LiteLLM model to verify (e.g., 'ollama/nomic-embed-text')",
    ),
    pull: bool = typer.Option(False, "--pull", help="Pull the Ollama model if not already available"),
):
    """
    Check that the environment is ready for benchmarking.
    """
    import importlib
    import subprocess

    indexer_deps = {
        "faiss": "faiss",
        "chroma": "chromadb",
        "qdrant": "qdrant_client",
        "milvus": "pymilvus",
        "lance": "lancedb",
        "weaviate": "weaviate",
        "duckdb": "duckdb",
        "usearch": "usearch",
        "annoy": "annoy",
        "hnswlib": "hnswlib",
        "scann": "scann",
        "vespa": "vespa",
        "elasticsearch": "elasticsearch",
        "pgvector": "psycopg2",
    }

    console.print("\n[bold cyan]Embenx Environment Check[/bold cyan]\n")

    # --- Indexers ---
    console.print("[bold]Indexers:[/bold]")
    all_ok = True
    for name, pkg in indexer_deps.items():
        try:
            importlib.import_module(pkg)
            console.print(f"  [green]✓[/green] {name}")
        except ImportError:
            console.print(f"  [yellow]✗[/yellow] {name} — [dim]uv pip install {pkg}[/dim]")
            all_ok = False

    # --- Ollama ---
    if model.startswith("ollama/"):
        model_name = model.split("/", 1)[1]
        console.print(f"\n[bold]Ollama ({model_name}):[/bold]")
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if model_name in result.stdout:
                console.print(f"  [green]✓[/green] Model '{model_name}' is available")
            elif pull:
                console.print(f"  [cyan]→[/cyan] Pulling {model_name}...")
                subprocess.run(["ollama", "pull", model_name], check=True)
                console.print(f"  [green]✓[/green] Pulled successfully")
            else:
                console.print(
                    f"  [yellow]✗[/yellow] Model not found. [dim]ollama pull {model_name}[/dim]"
                )
                all_ok = False
        except Exception as e:
            console.print(f"  [red]✗[/red] Ollama error: {e}")
            all_ok = False

    console.print(f"\n{'[bold green]✓ Ready!' if all_ok else '[bold yellow]⚠ Fix issues above.'}")


@app.command()
def mcp_start():
    """
    Start the Embenx MCP server for agentic tool-use.
    """
    import asyncio
    from mcp_server import run
    
    console.print("[bold green]Starting Embenx MCP Server...[/bold green]")
    console.print("[cyan]Connect your agent (Claude Desktop, etc.) via stdio.[/cyan]")
    asyncio.run(run())

@app.command()
def list_indexers():
    """
    List available indexing libraries for benchmarking.
    """
    from indexers import get_indexer_map

    console.print("[bold cyan]Available Indexers:[/bold cyan]")
    indexers = list(get_indexer_map().keys())
    for idx in indexers:
        console.print(f" - {idx}")


if __name__ == "__main__":
    app()
