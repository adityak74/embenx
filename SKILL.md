# Embenx Skill 🚀

This skill enables the agent to perform high-performance benchmarking of vector indexing libraries using the **Embenx** CLI.

## <instructions>
- **Objective**: Assist the user in evaluating and comparing vector search backends (FAISS, Chroma, Qdrant, Milvus, LanceDB).
- **Core Workflow**:
    1. **Identify Requirements**: Determine the user's dataset (HuggingFace name or local path) and the indexing libraries they wish to compare.
    2. **Check Availability**: Always run `uv run embenx list-indexers` first to see which backends are supported in the current environment.
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
