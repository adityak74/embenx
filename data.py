from datasets import load_dataset
from typing import List, Dict

def load_documents(dataset_name: str, split: str, text_column: str, max_docs: int) -> List[Dict]:
    """
    Load documents from a HuggingFace dataset.
    Returns a list of dicts: [{"id": idx, "text": "...", "metadata": {...}}]
    """
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")
    
    docs = []
    # If there are fewer docs than requested, adjust
    total_to_load = min(max_docs, len(ds))
    
    for i in range(total_to_load):
        row = ds[i]
        if text_column not in row:
            raise ValueError(f"Column '{text_column}' not found in dataset.")
            
        text = str(row[text_column])
        metadata = {k: v for k, v in row.items() if k != text_column}
        
        docs.append({
            "id": str(i),
            "text": text,
            "metadata": metadata
        })
        
    return docs
