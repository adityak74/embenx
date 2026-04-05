import json
import os
from typing import Any, Dict, List

from core import Collection

# --- Retrieval Zoo Mapping ---
# In a real scenario, these would point to Hugging Face datasets or S3 buckets.
ZOO_MAP = {
    "squad-v2": "https://huggingface.co/datasets/adityak74/embenx-zoo/resolve/main/squad-v2.parquet",
    "natural-questions": "https://huggingface.co/datasets/adityak74/embenx-zoo/resolve/main/nq.parquet",
    "ms-marco": "https://huggingface.co/datasets/adityak74/embenx-zoo/resolve/main/msmarco.parquet",
}

def load_from_zoo(dataset_name: str, cache_dir: str = ".embenx_cache") -> Collection:
    """
    Download and load a pre-built collection from the Embenx Retrieval Zoo.
    """
    if dataset_name not in ZOO_MAP:
        raise ValueError(f"Dataset '{dataset_name}' not found in zoo. Available: {list(ZOO_MAP.keys())}")
        
    url = ZOO_MAP[dataset_name]
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{dataset_name}.parquet")
    
    if not os.path.exists(local_path):
        import requests
        print(f"Downloading {dataset_name} from Embenx Zoo...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    return Collection.from_parquet(local_path)

def list_zoo() -> list:
    """List all available pre-built collections in the zoo."""
    return list(ZOO_MAP.keys())

def load_documents(
    dataset_name: str, subset: str = "default", split: str = "train", max_docs: int = 100
) -> List[Dict[str, Any]]:
    """
    Load documents from Hugging Face or local files.
    """
    if os.path.exists(dataset_name):
        # Local file path
        if dataset_name.endswith(".json"):
            with open(dataset_name, "r") as f:
                data = json.load(f)
                docs = data if isinstance(data, list) else [data]
        elif dataset_name.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(dataset_name)
            docs = df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported file format: {dataset_name}")
        return docs[:max_docs]

    # Hugging Face fallback
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, subset, split=split, streaming=True)
        docs = []
        for i, doc in enumerate(ds):
            if i >= max_docs:
                break
            docs.append(doc)
        return docs
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")

def save_collection(collection: Collection, path: str):
    """
    Save a collection's vectors and metadata to disk.
    """
    collection.to_parquet(path)
