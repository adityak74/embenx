# Embenx 🚀

[![GitHub stars](https://img.shields.io/github/stars/adityak74/embenx.svg?style=flat-square)](https://github.com/adityak74/embenx/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/adityak74/embenx.svg?style=flat-square)](https://github.com/adityak74/embenx/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![Ollama Support](https://img.shields.io/badge/Ollama-Supported-orange.svg?style=flat-square)](https://ollama.com/)

**Embenx** is a minimal, ultra-fast, and flexible CLI for benchmarking vector indexing libraries. Compare **FAISS, Chroma, Qdrant, Milvus, and LanceDB** using your own data or HuggingFace datasets with local or cloud-based embeddings.

[Explore Documentation](docs/index.html) · [Report Bug](https://github.com/adityak74/embenx/issues) · [Request Feature](https://github.com/adityak74/embenx/issues)

---

## 🌟 Key Features

- **⚡ Universal LLM Support**: Powered by `LiteLLM`—seamlessly works with Ollama (local), OpenAI, Anthropic, and more.
- **📦 Instant Datasets**: Direct integration with HuggingFace `datasets` (supports remote repositories and local CSV/JSON/JSONL).
- **📊 Comprehensive Metrics**: Track **Build Time**, **Query Latency**, **Index Footprint**, and **Memory Overhead** in a beautiful terminal dashboard.
- **🛠 Modular Architecture**: Easily evaluate and swap indexing backends to find the perfect fit for your production needs.

## 📋 Prerequisites

- **Python 3.9+**
- **Ollama** (Optional, for zero-cost local embeddings): [Install Ollama](https://ollama.com/) and pull a model:
  ```bash
  ollama pull nomic-embed-text
  ```
- **Cloud API Keys** (Optional): Set `OPENAI_API_KEY` or other provider keys for cloud-based embeddings.

## 🚀 Quick Start

### 1. Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
# Clone the repository
git clone https://github.com/adityak74/embenx.git
cd embenx

# Install and create virtual environment in one go
uv sync

# (Optional) Install Milvus-lite for local SQLite-based benchmarking
uv pip install milvus-lite
```

Alternatively, using standard pip:
```bash
pip install .
```

### 2. List Supported Indexers
```bash
# After 'uv sync', use the built-in command
uv run embenx list-indexers

# Or use the python file directly
python3 cli.py list-indexers
```

### 3. Run Your First Benchmark
Compare all indexers using the SQuAD dataset via local Ollama:

```bash
uv run embenx benchmark --dataset squad --max-docs 100 --model ollama/nomic-embed-text
```

## 🛠 Advanced Usage

### Local Data Benchmarking
Benchmark your own private data files (CSV, JSON, JSONL):
```bash
uv run embenx benchmark --dataset json --data-files ./my_data.jsonl --text-column content
```

### High-Dimensional Comparison
Evaluate how different models (e.g., 768 vs 4096 dimensions) impact indexing performance:
```bash
python3 cli.py benchmark --dataset squad --max-docs 10 --model ollama/qwen3-embedding
```

## ⚙️ Parameters

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--dataset` / `-d` | HuggingFace dataset name or format (`csv`, `json`). | *Required* |
| `--text-column` / `-c` | The column to embed and index. | `text` |
| `--max-docs` / `-m` | Number of documents to index. | `1000` |
| `--indexers` / `-i` | Comma-separated list (`faiss,chroma,qdrant,lance,milvus`) or `all`. | `all` |
| `--model` | LiteLLM model string. | `ollama/nomic-embed-text` |
| `--data-files` | Path to local data files. | `None` |

## 📊 Output Metrics

| Metric | What it measures |
| :--- | :--- |
| **Build Time (s)** | Total time to embed documents and build the index structure. |
| **Query Time (ms/query)** | Average search latency over 10 test queries. |
| **Index Size (KB)** | Estimated memory or disk footprint of the final index. |
| **Memory Added (MB)** | Real-time change in process RAM usage during the indexing phase. |

## 🤝 Contributing

Contributions are welcome! Whether it's adding a new indexer, improving metrics, or fixing bugs, please feel free to open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
Built with ❤️ for the AI engineering community by [adityak74](https://github.com/adityak74).
