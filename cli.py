import typer
from rich.console import Console

app = typer.Typer(help="Embenx 🚀: Benchmark vector indexing libraries with ease")
console = Console()

@app.command()
def benchmark(
    dataset: str = typer.Option(..., "--dataset", "-d", help="HuggingFace dataset name or data format (e.g., 'csv', 'json')"),
    split: str = typer.Option("train", "--split", "-s", help="Dataset split to use"),
    text_column: str = typer.Option("text", "--text-column", "-c", help="Column containing the text to index"),
    max_docs: int = typer.Option(1000, "--max-docs", "-m", help="Maximum number of documents to index"),
    indexers: str = typer.Option("all", "--indexers", "-i", help="Comma-separated list (e.g., faiss,chroma,qdrant,milvus,lance) or 'all'"),
    model: str = typer.Option("ollama/nomic-embed-text", "--model", help="LiteLLM model name (e.g., 'ollama/nomic-embed-text', 'openai/text-embedding-3-small')"),
    data_files: str = typer.Option(None, "--data-files", help="Path to local data files (for CSV/JSON formats)"),
    cleanup: bool = typer.Option(True, "--cleanup/--no-cleanup", help="Automatically cleanup temporary index files after benchmarking")
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
    console.print(f"Max Docs: [cyan]{max_docs}[/cyan]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"Indexers: [cyan]{indexers}[/cyan]")
    
    # We will import the benchmark engine here to avoid early loading overhead
    from benchmark import run_benchmark
    
    # Parse indexers
    if indexers.lower() == "all":
        selected_indexers = ["faiss", "chroma", "qdrant", "milvus", "lance"]
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
        cleanup=cleanup
    )

@app.command()
def init_skill():
    """
    Generate a SKILL.md file for AI agents (Claude, Gemini, etc.) to use Embenx.
    """
    import os
    
    skill_content = """# Embenx Skill 🚀

This skill enables the agent to perform high-performance benchmarking of vector indexing libraries using the **Embenx** CLI.

## <instructions>
- **Objective**: Assist the user in evaluating and comparing vector search backends (FAISS, Chroma, Qdrant, Milvus, LanceDB).
- **Core Workflow**:
    1. **Identify Requirements**: Determine the user's dataset (HuggingFace name or local path) and the indexing libraries they wish to compare.
    2. **Check Environment**: Always run `uv run embenx setup` first — it verifies installed indexers and checks the embedding model is ready. Pass `--pull` to auto-pull a missing Ollama model.
    3. **Execute Benchmark**:
        - For standard HF datasets: `uv run embenx benchmark --dataset <name> --max-docs <num>`.
        - For local data: `uv run embenx benchmark --dataset json --data-files <path> --text-column <col>`.
        - To compare embedding models: Vary the `--model` flag (e.g., `ollama/nomic-embed-text` vs `openai/text-embedding-3-small`).
    4. **Analyze Results**: Review the generated table (Build Time, Query Latency, Index Size, Memory) and provide a technical recommendation based on the user's constraints (e.g., "Choose Qdrant for fastest query latency with low memory").
    5. **Cleanup**: After benchmarking, run `uv run embenx cleanup` to remove temporary database files unless the user specifically asked to keep them for inspection.
- **Safety**:
    - Ensure Ollama is running if local models are requested.
    - Verify local data paths exist before running the benchmark.
    - Default to a small `--max-docs` (e.g., 100) for initial testing to save time/tokens.
</instructions>

## <available_resources>
- **Documentation**: `docs/index.html` (Module reference).
- **CLI Help**: `uv run embenx help` or `uv run embenx <command> --help`.
- **Examples**: `examples/ollama_benchmark.sh`.
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
def help(ctx: typer.Context):
    """
    Display the help menu for Embenx.
    """
    typer.echo(ctx.parent.get_help())

@app.command()
def cleanup():
    """
    Manually remove any leftover benchmark artifacts (*.db, *.lance, etc.)
    """
    import os
    import shutil
    import glob
    
    console.print("[bold yellow]Cleaning up benchmark artifacts...[/bold yellow]")
    
    patterns = ["*.db", "*.lance", "benchmark", "benchmark.lance"]
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
    model: str = typer.Option("ollama/nomic-embed-text", "--model", help="LiteLLM model to verify (e.g., 'ollama/nomic-embed-text')"),
    pull: bool = typer.Option(False, "--pull", help="Pull the Ollama model if not already available"),
):
    """
    Check that the environment is ready for benchmarking.

    Verifies installed indexers and (for Ollama models) that the Ollama
    server is reachable and the requested model is available.
    """
    import importlib

    indexer_deps = {
        "faiss": "faiss",
        "chroma": "chromadb",
        "qdrant": "qdrant_client",
        "milvus": "pymilvus",
        "lance": "lancedb",
    }

    console.print("\n[bold cyan]Embenx Environment Check[/bold cyan]\n")

    # --- Indexers ---
    console.print("[bold]Indexers:[/bold]")
    all_ok = True
    for name, pkg in indexer_deps.items():
        try:
            importlib.import_module(pkg)
            console.print(f"  [green]✓[/green] {name} ({pkg})")
        except ImportError:
            console.print(f"  [yellow]✗[/yellow] {name} — not installed  [dim](uv pip install {pkg})[/dim]")
            all_ok = False

    # --- Ollama ---
    if model.startswith("ollama/"):
        model_name = model.split("/", 1)[1]
        console.print(f"\n[bold]Ollama ({model_name}):[/bold]")
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                raise RuntimeError("ollama list failed")

            available_models = result.stdout
            if model_name in available_models:
                console.print(f"  [green]✓[/green] Ollama is running and '{model_name}' is available")
            else:
                console.print(f"  [yellow]✗[/yellow] Ollama is running but '{model_name}' is not pulled")
                if pull:
                    console.print(f"  [cyan]→[/cyan] Pulling {model_name}...")
                    subprocess.run(["ollama", "pull", model_name], check=True)
                    console.print(f"  [green]✓[/green] '{model_name}' pulled successfully")
                else:
                    console.print(f"  [dim]  Run: ollama pull {model_name}  (or pass --pull)[/dim]")
                    all_ok = False
        except FileNotFoundError:
            console.print("  [red]✗[/red] Ollama is not installed or not in PATH")
            console.print("  [dim]  Install from https://ollama.com[/dim]")
            all_ok = False
        except Exception as e:
            console.print(f"  [red]✗[/red] Could not reach Ollama: {e}")
            all_ok = False

    console.print()
    if all_ok:
        console.print("[bold green]✓ Environment is ready for benchmarking.[/bold green]")
    else:
        console.print("[bold yellow]⚠ Fix the issues above before running a benchmark.[/bold yellow]")


@app.command()
def list_indexers():
    """
    List available indexing libraries for benchmarking.
    """
    console.print("[bold cyan]Available Indexers:[/bold cyan]")
    indexers = ["faiss", "chroma", "qdrant", "milvus", "lance"]
    for idx in indexers:
        console.print(f" - {idx}")

if __name__ == "__main__":
    app()
