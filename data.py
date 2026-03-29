import os
from typing import Dict, List

from datasets import load_dataset


def load_documents(
    dataset_name: str, split: str, text_column: str, max_docs: int, data_files: str = None
) -> List[Dict]:
    """
    Load documents from a HuggingFace dataset or local files.
    Returns a list of dicts: [{"id": idx, "text": "...", "metadata": {...}}]
    """
    # If dataset_name is a local file path, determine the format
    if os.path.exists(dataset_name) and data_files is None:
        if dataset_name.endswith(".parquet"):
            data_files = dataset_name
            dataset_name = "parquet"
        elif dataset_name.endswith(".csv"):
            data_files = dataset_name
            dataset_name = "csv"
        elif dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
            data_files = dataset_name
            dataset_name = "json"

    try:
        ds = load_dataset(dataset_name, data_files=data_files, split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}") from e

    # Efficiently slice the dataset
    subset = ds.select(range(min(max_docs, len(ds))))

    if text_column not in subset.column_names:
        raise ValueError(
            f"Column '{text_column}' not found in dataset. Available: {subset.column_names}"
        )

    docs = []
    for i, row in enumerate(subset):
        text = str(row.pop(text_column))
        docs.append({"id": str(i), "text": text, "metadata": row})

    return docs
