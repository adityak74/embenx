#!/bin/bash

# Example script to run a benchmark using local Ollama.
# Ensure Ollama is running and the model is pulled:
# ollama pull nomic-embed-text

DATASET="squad"
TEXT_COLUMN="question"
MAX_DOCS=100
MODEL="ollama/nomic-embed-text"
INDEXERS="faiss,lancedb,chroma"

echo "🚀 Starting Embenx benchmark with model: $MODEL..."

python3 cli.py benchmark \
    --dataset "$DATASET" \
    --text-column "$TEXT_COLUMN" \
    --max-docs "$MAX_DOCS" \
    --indexers "$INDEXERS" \
    --model "$MODEL"

if [ $? -eq 0 ]; then
    echo "✅ Benchmark completed successfully."
else
    echo "❌ Benchmark failed. Check if Ollama is running and the model is pulled."
    exit 1
fi
