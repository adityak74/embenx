import numpy as np
from datasets import load_dataset

from core import Collection
from llm import Embedder


def seed_real_data(
    dataset_name: str = "ag_news",
    model_name: str = "ollama/nomic-embed-text",
    limit: int = 400,
    use_random: bool = False,
):
    print(f"🚀 Loading real dataset: {dataset_name}...")

    # Load dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)

    docs = []
    categories = ["World", "Sports", "Business", "Sci/Tech"]

    # Group by category to get an even split
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    per_cat = limit // 4

    for item in ds:
        label = item["label"]
        if counts[label] < per_cat:
            docs.append({"text": item["text"], "label": categories[label], "id": len(docs)})
            counts[label] += 1

        if len(docs) >= limit:
            break

    texts = [d["text"] for d in docs]

    if use_random:
        print("⚠️ Using random vectors for demo (model not available)...")
        dim = 768
        vectors = []
        # Create cluster centers to make it look realistic
        centers = [np.random.randn(dim) * 2 for _ in range(4)]
        for doc in docs:
            label_idx = categories.index(doc["label"])
            vec = centers[label_idx] + np.random.randn(dim) * 0.5
            vectors.append(vec.tolist())
    else:
        print(f"🧠 Embedding {len(texts)} texts using {model_name}...")
        embedder = Embedder(model_name=model_name)
        vectors = embedder.embed_texts(texts)

        if not vectors:
            print("❌ Embedding failed. Falling back to random vectors.")
            dim = 768
            vectors = np.random.randn(len(texts), dim).tolist()
        else:
            dim = len(vectors[0])

    # Setup a collection
    col = Collection(name="ag_news_explorer", dimension=dim)

    # Add to collection
    print(f"📥 Adding {len(vectors)} documents to collection...")
    col.add(vectors, docs)

    # Save to Parquet so Explorer can see it
    col.to_parquet("ag_news_explorer.parquet")
    print(f"✅ Successfully created 'ag_news_explorer.parquet' with {len(docs)} documents.")
    print("👉 Run 'embenx explorer' and select 'ag_news_explorer' to visualize.")


if __name__ == "__main__":
    import sys

    # Default to random if no args provided to ensure it works without ollama setup
    random_mode = "--random" in sys.argv
    seed_real_data(use_random=random_mode)
