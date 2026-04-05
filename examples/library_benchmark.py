from embenx.benchmark import generate_report, run_benchmark
from rich.console import Console


def run_report_example():
    console = Console()
    print("--- Library-Native Benchmark & Report Example ---")
    
    # 1. Run a benchmark on a dummy dataset
    print("Benchmarking indexers on 'dummy' dataset...")
    results = run_benchmark(
        dataset_name="dummy",
        split="train",
        text_column="text",
        max_docs=50,
        indexer_names=["faiss", "simple"],
        model_name="ollama/nomic-embed-text",
        console=console
    )
    
    # 2. Generate a Markdown technical report
    if results:
        report_path = generate_report(results, "Dummy Dataset")
        print(f"\nTechnical Report generated: {report_path}")
        
        with open(report_path, "r") as f:
            print("\n--- Report Preview ---")
            print(f.read()[:200] + "...")

if __name__ == "__main__":
    run_report_example()
