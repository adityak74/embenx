#!/bin/bash

# Example script for benchmarking a local Parquet file

# 1. Create a sample parquet file if it doesn't exist
python << EOF
import pandas as pd
df = pd.DataFrame({
    "content": [
        "Vector databases are essential for modern AI applications.",
        "Embenx makes it easy to compare different vector libraries.",
        "Sphinx is a great tool for generating documentation.",
        "Parquet is a columnar storage file format optimized for data processing.",
        "LiteLLM provides a unified interface for various LLM providers."
    ],
    "category": ["AI", "Tools", "Docs", "Data", "LLM"]
})
df.to_parquet("data.parquet")
print("Created data.parquet")
EOF

# 2. Run the benchmark using the direct file path
# We specify --text-column because our data uses 'content' instead of the default 'text'
uv run embenx benchmark \
    --dataset data.parquet \
    --text-column content \
    --max-docs 10 \
    --indexers faiss,duckdb

# 3. Cleanup
rm data.parquet
