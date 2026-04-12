import time

import numpy as np

from embenx.core import TemporalCollection


def run_echo_example():
    print("--- Echo Temporal Episodic Memory Example ---")

    # 1. Setup a TemporalCollection
    dim = 64
    col = TemporalCollection(name="chat_history_memory", dimension=dim)

    # 2. Add memories with different timestamps
    print("Adding memories with different timestamps...")
    now = time.time()
    vectors = np.random.rand(3, dim).astype(np.float32)

    # Memory 1: 1 hour ago
    # Memory 2: 1 day ago
    # Memory 3: Now
    timestamps = [now - 3600, now - 86400, now]
    metadata = [
        {"id": "msg_1h", "text": "Discussion about project scope."},
        {"id": "msg_1d", "text": "Initial greeting and introductions."},
        {"id": "msg_now", "text": "Finalizing the implementation details."},
    ]

    col.add_temporal(vectors, timestamps, metadata)

    # 3. Perform recency-biased search
    query = vectors[0]  # Similar to the 1 hour ago message
    print("\n1. Searching with 50% recency weight...")
    results = col.search_temporal(query, top_k=2, recency_weight=0.5)
    for i, (meta, score) in enumerate(results):
        print(
            f" {i+1}. {meta['text']} (Combined Dist: {score:.4f}, Timestamp: {meta['timestamp']})"
        )

    # 4. Search with a specific time window (last 2 hours)
    print("\n2. Searching within a 2-hour time window...")
    window = (now - 7200, now + 10)
    results_window = col.search_temporal(query, top_k=5, time_window=window)
    for i, (meta, score) in enumerate(results_window):
        print(f" {i+1}. {meta['text']} (In Window: {meta['id']})")


if __name__ == "__main__":
    run_echo_example()
