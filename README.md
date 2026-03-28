<div align="center">

<h1>Embenx 🚀</h1>

<p>
  <strong>Minimal, ultra-fast CLI for benchmarking vector indexing libraries.</strong><br/>
  Compare FAISS, Chroma, Qdrant, Milvus, and LanceDB with any dataset — local or HuggingFace.
</p>

<p>
  <a href="https://github.com/adityak74/embenx/stargazers"><img src="https://img.shields.io/github/stars/adityak74/embenx?style=flat-square&color=yellow" alt="Stars"/></a>
  <a href="https://github.com/adityak74/embenx/issues"><img src="https://img.shields.io/github/issues/adityak74/embenx?style=flat-square" alt="Issues"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="MIT License"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square" alt="Python 3.9+"/></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-supported-orange.svg?style=flat-square" alt="Ollama Support"/></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/uv-ready-purple.svg?style=flat-square" alt="uv ready"/></a>
</p>

<p>
  <a href="docs/index.html">Documentation</a> ·
  <a href="https://github.com/adityak74/embenx/issues">Report Bug</a> ·
  <a href="https://github.com/adityak74/embenx/issues">Request Feature</a>
</p>

</div>

---

## What is Embenx?

Embenx is a single-command benchmarking tool for vector search backends. Point it at any HuggingFace dataset or local file, pick your embedding model, and get a side-by-side comparison of build time, query latency, index size, and memory usage across all major vector libraries.

```
Indexer   Build(s)  Query(ms)  Index(KB)  Memory(MB)
--------  --------  ---------  ---------  ----------
faiss         0.42       0.31        580        18.4
chroma        1.10       3.21       1200        41.0
qdrant        2.31       1.94       4800        64.2
lance         0.55       2.10        310        22.0
milvus        3.74       2.60       2400        87.0
```

## Features

- **Universal model support** — Ollama (local, free), OpenAI, Anthropic, and any LiteLLM provider
- **Any dataset** — HuggingFace datasets or local CSV / JSON / JSONL files
- **Four metrics** — build time, query latency, index footprint, memory overhead
- **Environment setup** — `embenx setup` checks all dependencies in one command
- **AI-agent ready** — ships with `embenx init-skill` to generate a `SKILL.md` for Claude and other agents

## Installation

**Recommended (uv):**
```bash
git clone https://github.com/adityak74/embenx.git
cd embenx
uv sync
```

**pip:**
```bash
pip install .
```

**Optional — Milvus support:**
```bash
uv pip install milvus-lite
```

## Quick Start

### 1. Check your environment
```bash
uv run embenx setup
```
This verifies all indexers are installed and your Ollama model is ready. Pass `--pull` to auto-download a missing model.

### 2. Run a benchmark
```bash
uv run embenx benchmark --dataset squad --max-docs 100 --model ollama/nomic-embed-text
```

### 3. Benchmark your own data
```bash
uv run embenx benchmark \
  --dataset json \
  --data-files ./my_data.jsonl \
  --text-column content \
  --max-docs 500
```

### 4. Compare embedding models
```bash
# Run twice with different models to compare dimension/speed tradeoffs
uv run embenx benchmark --dataset squad --max-docs 200 --model ollama/nomic-embed-text
uv run embenx benchmark --dataset squad --max-docs 200 --model ollama/qwen3-embedding
```

## CLI Reference

### `embenx benchmark`

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--dataset` / `-d` | HuggingFace dataset name or `csv` / `json` | *required* |
| `--text-column` / `-c` | Column to embed and index | `text` |
| `--split` / `-s` | HuggingFace dataset split | `train` |
| `--max-docs` / `-m` | Number of documents to index | `1000` |
| `--indexers` / `-i` | Comma-separated list or `all` | `all` |
| `--model` | LiteLLM model string | `ollama/nomic-embed-text` |
| `--data-files` | Path to local file(s) | — |
| `--no-cleanup` | Keep index files on disk after the run | — |

### Other commands

| Command | Description |
| :--- | :--- |
| `embenx setup` | Check installed indexers and embedding model availability |
| `embenx setup --pull` | Same, and auto-pull a missing Ollama model |
| `embenx list-indexers` | List supported backends |
| `embenx cleanup` | Remove leftover index artifacts |
| `embenx init-skill` | Generate a `SKILL.md` for AI agents |
| `embenx help` | Show help menu |

## Output Metrics

| Metric | What it measures |
| :--- | :--- |
| **Build Time (s)** | Time to embed all documents and construct the index |
| **Query Time (ms/query)** | Mean search latency across 10 test queries |
| **Index Size (KB)** | Disk or memory footprint of the final index |
| **Memory Added (MB)** | Process RAM delta during the indexing phase |

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Ollama](https://ollama.com/) for local, zero-cost embeddings (optional)
- Cloud API key (`OPENAI_API_KEY`, etc.) for cloud embeddings (optional)

## Contributing

Contributions are welcome and appreciated! Here's how to get started:

1. **Fork** the repository and create a branch: `git checkout -b feature/my-feature`
2. **Install** dependencies: `uv sync`
3. **Make your changes** — new indexers, metrics, bug fixes, or docs
4. **Test** your changes: `uv run embenx benchmark --dataset squad --max-docs 50`
5. **Open a Pull Request** with a clear description of what changed and why

### Ideas for contributions

- Add a new vector indexer backend (e.g., Weaviate, Pinecone, pgvector)
- Add new output metrics (e.g., recall@k, P95 latency)
- Improve the terminal dashboard output
- Add more example scripts under `examples/`
- Improve documentation

If you're unsure whether something is a good fit, [open an issue](https://github.com/adityak74/embenx/issues) first and we can discuss it.

## License

Distributed under the **MIT License** — free to use, modify, and distribute with attribution.

See [`LICENSE`](LICENSE) for the full license text.

---

<div align="center">
  Built with ❤️ for the AI engineering community by <a href="https://github.com/adityak74">adityak74</a>
</div>
