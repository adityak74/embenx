import numpy as np

from embenx.core import AgenticCollection


def run_agentic_example():
    print("--- Agentic Self-Healing Memory Example ---")

    # 1. Setup an AgenticCollection
    dim = 64
    col = AgenticCollection(name="autonomous_brain", dimension=dim)

    # 2. Add some ambiguous data
    print("Adding ambiguous memories...")
    vectors = np.array(
        [
            [1, 0, 0, 0],
            [0.95, 0.05, 0, 0],
            [0.9, 0.1, 0, 0],
        ]
    ).astype(np.float32)
    # Pad to dim
    vectors = np.pad(vectors, ((0, 0), (0, dim - 4)))

    metadata = [
        {"id": "doc_a", "text": "Very relevant info"},
        {"id": "doc_b", "text": "Slightly relevant info"},
        {"id": "doc_c", "text": "Noise result"},
    ]

    col.add(vectors, metadata)

    # 3. Initial Search
    print("\nInitial search results (Pure semantic):")
    query = vectors[0]
    results = col.search(query, top_k=3)
    for meta, dist in results:
        print(f" - {meta['id']}: {meta['text']} (Dist: {dist:.4f})")

    # 4. Provide Feedback (Self-Healing)
    print("\nMarking 'doc_c' as 'bad' and 'doc_b' as 'good'...")
    col.feedback("doc_c", label="bad")
    col.feedback("doc_b", label="good")
    col.feedback("doc_b", label="good")  # Double boost

    # 5. Agentic Search (with feedback boost)
    print("\nAgentic search results (Incorporating feedback):")
    results_agentic = col.agentic_search(query, top_k=3)
    for meta, score in results_agentic:
        # Note: Score here is adjusted distance
        fb = meta.get("feedback_score", 0.0)
        print(f" - {meta['id']}: {meta['text']} (Adjusted Score: {score:.4f}, FB: {fb:.1f})")


if __name__ == "__main__":
    run_agentic_example()
