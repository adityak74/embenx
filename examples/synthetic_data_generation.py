import os
from unittest.mock import patch

import numpy as np

from core import Collection


def run_synthetic_data_example():
    """
    Example demonstrating how to generate synthetic query-document pairs
    from an Embenx collection for fine-tuning or evaluation.
    """
    print("--- Embenx Synthetic Data Generation Example ---")

    # 1. Initialize a collection and add some example documents
    col = Collection(dimension=4, name="news_cluster")

    # Simple semantic data
    vectors = np.random.rand(5, 4).astype(np.float32)
    metadata = [
        {"id": 1, "text": "New advances in quantum computing were announced today in Zurich."},
        {"id": 2, "text": "The local football team won the championship after a thrilling final."},
        {"id": 3, "text": "Electric vehicle sales have reached a record high this quarter."},
        {
            "id": 4,
            "text": "A new species of deep-sea jellyfish was discovered in the Pacific Ocean.",
        },
        {
            "id": 5,
            "text": "Stock markets showed volatility following the latest central bank report.",
        },
    ]

    col.add(vectors, metadata)
    print(f"Added {len(metadata)} documents to the '{col.name}' collection.")

    # 2. Generate synthetic search queries using an LLM
    # In a real scenario, this would call LiteLLM (v1.83.0+) to hit GPT-4, Claude, or Ollama.
    # Here, we'll mock the response for demonstration if no API key is found.

    print("\nGenerating synthetic search queries for the indexed data...")

    # Check for API key to decide whether to mock
    has_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    if not has_api_key:
        print("[Note: No LLM API key found. Mocking the generation for demonstration.]")

        # Global mock for this example
        class MockMessage:
            content = "What are the latest breakthroughs in quantum physics?\nHow is the EV market performing lately?"

        class MockChoice:
            message = MockMessage()

        class MockResponse:
            choices = [MockChoice()]

        patcher = patch("litellm.completion", return_value=MockResponse())
        patcher.start()

    # 2. Generate synthetic search queries using an LLM
    print("\nGenerating synthetic search queries for the indexed data...")
    # Tip: To use a local Ollama model, use:
    # results = col.generate_synthetic_queries(model="ollama/llama3", api_base="http://localhost:11434")
    results = col.generate_synthetic_queries(
        text_key="text", n_queries_per_doc=2, num_docs=2, model="gpt-4o-mini"
    )

    # 3. Inspect the generated synthetic dataset
    print(f"\nSuccessfully generated {len(results)} query-document pairs:")
    for i, item in enumerate(results):
        print(f"\n[{i+1}] Query: \"{item['query']}\"")
        print(f"    Source Doc ID: {item['doc_id']}")
        print(f"    Snippet: {item['doc_text'][:60]}...")

    # 4. Exporting the dataset for training/evaluation
    output_file = "synthetic_eval_data.jsonl"
    col.generate_synthetic_queries(
        text_key="text", n_queries_per_doc=1, num_docs=5, output_path=output_file
    )

    if os.path.exists(output_file):
        print(f"\nSynthetic dataset exported to: {output_file}")
        # Clean up for the example
        os.remove(output_file)


if __name__ == "__main__":
    run_synthetic_data_example()
