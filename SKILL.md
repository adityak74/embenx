# Embenx Skill 🚀

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
