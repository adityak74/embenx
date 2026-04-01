from embenx import Collection
import numpy as np
import os

def run_multimodal_example():
    print("--- Multimodal Image Retrieval Example ---")
    
    # 1. Setup a collection for images
    # CLIP-style models usually have 512 or 768 dimensions
    col = Collection(dimension=512, indexer_type="faiss")
    
    # 2. Add images
    # In a real scenario, these would be real paths
    image_paths = ["cat.jpg", "dog.jpg", "car.png"]
    
    # We use mock data for this example to avoid requiring real image files
    print("Indexing images (simulated)...")
    mock_vectors = np.random.rand(3, 512).astype(np.float32)
    col.add(mock_vectors, metadata=[{"id": path, "label": path.split('.')[0]} for path in image_paths])

    # 3. Search with an image
    # In reality: results = col.search_image("query_cat.jpg")
    print("\nSearching for 'cat' using image-query (simulated)...")
    query_vec = mock_vectors[0]
    results = col.search(query_vec, top_k=1)
    
    for meta, dist in results:
        print(f" Found matching image: {meta['id']} (Label: {meta['label']}, Distance: {dist:.4f})")

if __name__ == "__main__":
    run_multimodal_example()
