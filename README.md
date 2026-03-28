# Embenx 🚀

A minimal, fast, and flexible Python CLI for benchmarking vector indexing libraries. Compare **FAISS, Chroma, Qdrant, Milvus, and LanceDB** using your own data or HuggingFace datasets with local or cloud-based embeddings.

---

## 🌟 Key Features
- **Universal LLM Support**: Powered by `LiteLLM`—seamlessly works with Ollama (local), OpenAI, Anthropic, and more.
- **Instant Datasets**: Direct integration with HuggingFace `datasets` (support for remote and local data).
- **Comprehensive Metrics**: Track build time, query latency, index footprint, and memory overhead in a beautiful terminal dashboard.
- **Modular & Fast**: Designed as a surgical tool for developers to evaluate search infrastructure.

## 📋 Prerequisites
- **Python 3.9+**
- **Ollama** (Optional, for local embeddings): [Install Ollama](https://ollama.com/) and pull a model:
  ```bash
  ollama pull nomic-embed-text
  ```
- **Cloud API Keys** (Optional): Set `OPENAI_API_KEY` or other provider keys for cloud-based embeddings.

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/adityak74/embenx.git
cd embenx

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install Milvus-lite for local benchmarking
pip install milvus-lite
```

## 🛠 Usage

### 1. List Supported Indexers
```bash
python3 cli.py list-indexers
```

### 2. Run a Benchmark
The CLI uses the [LiteLLM model format](https://docs.litellm.ai/docs/embedding/supported_embedding).

```bash
# Benchmark all indexers using the SQuAD dataset via local Ollama
python3 cli.py benchmark --dataset squad --max-docs 100 --model ollama/nomic-embed-text

# Compare performance with high-dimensional models
python3 cli.py benchmark --dataset squad --max-docs 10 --model ollama/qwen3-embedding

# Use local data (e.g., a local JSONL file)
python3 cli.py benchmark --dataset json --data-files my_data.jsonl --text-column content
```

### ⚙️ Parameters
- `--dataset` / `-d`: HuggingFace dataset name or a data format (e.g., `csv`, `json`).
- `--text-column` / `-c`: The column to embed (default: `text`).
- `--max-docs` / `-m`: Number of documents to index (default: `1000`).
- `--indexers` / `-i`: Comma-separated list (`faiss,chroma,qdrant,lance,milvus`) or `all`.
- `--model`: LiteLLM model string (default: `ollama/nomic-embed-text`).

## 📊 Output Metrics
The tool generates a comparative table with the following metrics:
- **Build Time (s)**: The total time to embed documents and build the index.
- **Query Time (ms/query)**: Average search latency over a representative set of queries.
- **Index Size (KB)**: The estimated memory or disk footprint of the index.
- **Memory Added (MB)**: The actual change in process memory usage during indexing.

## 📚 Documentation & Examples
- **HTML Reference**: Open `docs/index.html` in your browser for the full API documentation.
- **Shell Example**: Run `./examples/ollama_benchmark.sh` for a quick demonstration.

---
Built with ❤️ for the AI engineering community.
