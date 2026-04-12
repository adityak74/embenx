import numpy as np

from embenx.core import SpatialCollection


def run_spatial_example():
    print("--- ESWM Neuroscience-inspired Spatial Memory Example ---")

    # 1. Setup a SpatialCollection
    dim = 64
    col = SpatialCollection(name="robot_spatial_memory", dimension=dim)

    # 2. Add episodic memories at different locations
    # Vectors: semantic content of what was seen
    # Coords: [x, y, z] location
    n = 10
    vectors = np.random.rand(n, dim).astype(np.float32)
    coords = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [5, 5, 0],
            [10, 10, 0],
            [2, 0, 0],
            [8, 8, 0],
            [1, 2, 0],
            [0, 5, 0],
            [6, 1, 0],
            [3, 3, 0],
        ]
    ).astype(np.float32)

    metadata = [{"id": i, "description": f"Observation at point {coords[i]}"} for i in range(n)]

    print("Indexing episodic memories with spatial anchors...")
    col.add_spatial(vectors, coords, metadata)

    # 3. Perform spatial-aware search
    # The agent is at [1, 1, 0] and remembers something similar to vectors[0]
    query = vectors[0]
    current_pos = np.array([1, 1, 0]).astype(np.float32)

    print(f"\nSearching for memories similar to 'query' within 5m of {current_pos}...")
    results = col.search_spatial(query, current_pos, top_k=3, spatial_radius=5.0)

    for i, (meta, score) in enumerate(results):
        print(f" {i+1}. {meta['description']} (Combined Score: {score:.4f})")


if __name__ == "__main__":
    run_spatial_example()
