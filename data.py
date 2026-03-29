from datasets import load_dataset
from typing import List, Dict

def load_documents(dataset_name: str, split: str, text_column: str, max_docs: int, data_files: str = None) -> List[Dict]:
    """
    Load documents from a HuggingFace dataset or local files.
    Returns a list of dicts: [{"id": idx, "text": "...", "metadata": {...}}]
    """
    try:
        ds = load_dataset(dataset_name, data_files=data_files, split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")
    
    # Efficiently slice the dataset
    subset = ds.select(range(min(max_docs, len(ds))))
    
    if text_column not in subset.column_names:
        raise ValueError(f"Column '{text_column}' not found in dataset. Available: {subset.column_names}")
    
    docs = []
    for i, row in enumerate(subset):
        text = str(row.pop(text_column))
        docs.append({
            "id": str(i),
            "text": text,
            "metadata": row
        })
        
    return docs
