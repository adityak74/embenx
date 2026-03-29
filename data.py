import os
from typing import Dict, List

import numpy as np
from datasets import load_dataset


def load_documents(
    dataset_name: str, split: str, text_column: str, max_docs: int, data_files: str = None
) -> List[Dict]:
    """
    Load documents from a HuggingFace dataset, local files, or NumPy arrays.
    Returns a list of dicts: [{"id": idx, "text": "...", "metadata": {...}}]
    """
    # Handle NumPy files
    if dataset_name.endswith(".npy") or dataset_name.endswith(".npz") or dataset_name.endswith(".index"):
        try:
            if dataset_name.endswith(".index"):
                # .index is a serialized FAISS index.
                # In this case, we return a special marker so benchmark knows to skip embedding.
                return [{"id": "serialized", "text": "", "metadata": {}, "index_path": dataset_name}]
            
            if dataset_name.endswith(".npy"):
                data = np.load(dataset_name)
                # For .npy, we assume it's an array of objects/strings if used as "text"
                # or we convert to list of strings
                docs = []
                for i in range(min(max_docs, len(data))):
                    docs.append({"id": str(i), "text": str(data[i]), "metadata": {}})
                return docs
            else:
                data = np.load(dataset_name, allow_pickle=True)
                # For .npz, look for specific keys
                vectors = data.get("vectors")
                texts = data.get("text") or data.get("texts")
                metadata = data.get("metadata")

                docs = []
                limit = min(max_docs, len(texts) if texts is not None else len(vectors))
                for i in range(limit):
                    doc = {"id": str(i)}
                    doc["text"] = str(texts[i]) if texts is not None else ""
                    doc["metadata"] = metadata[i] if metadata is not None else {}
                    docs.append(doc)
                return docs
        except Exception as e:
            raise RuntimeError(f"Failed to load NumPy file '{dataset_name}': {e}") from e

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
