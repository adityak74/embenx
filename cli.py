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
