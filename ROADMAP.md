# Embenx Roadmap 🚀

This document outlines the strategic direction of Embenx as it evolves from a benchmarking CLI into a **Python-native embedding retrieval layer**.

## Vision
Embenx aims to be the standard toolkit for Python developers to move embeddings and metadata through their pipelines, providing robust local retrieval, portable formats, and production-ready query features without the overhead of a full database cluster.

---

## Milestone 1: Core Retrieval & Library API (v0.1.0 - MVP)
**Goal**: Transition from CLI-only to a usable Python library with a stable API.

- [ ] **Unified Collection API**: Implement a `Collection` class to wrap indexers.
    - `Collection.add(vectors, metadata)`
    - `Collection.search(query, top_k)`
    - `Collection.save(path)` / `Collection.load(path)`
- [ ] **Refined I/O**:
    - `Collection.from_numpy(path)`
    - `Collection.from_parquet(path, vector_col, text_col)`
    - `Collection.from_faiss(path)`
- [ ] **Distance Backend Consistency**: Ensure L2, Cosine, and Dot Product work identically across all core indexers.
- [ ] **Batch Operations**: Support batch search for high-throughput pipelines.

## Milestone 2: Metadata & Filtering (v0.2.0)
**Goal**: Make retrieval production-aware with schemas and filters.

- [ ] **Schema Model**: Formalize the internal storage of `id`, `vector`, `text`, and `metadata`.
- [ ] **Filtering DSL**: Implement a `where` clause for metadata filtering.
    - Support both pre-filtering (for FAISS/NumPy) and native filtering (for Qdrant/LanceDB).
- [ ] **Parquet-Native Schema**: Ensure metadata travels with vectors in Parquet exports.

## Milestone 3: Hybrid Search & Reranking (v0.3.0)
**Goal**: Achieve state-of-the-art retrieval quality.

- [ ] **Sparse Vector Support**: Add BM25 or SPLADE integration.
- [ ] **Hybrid Search**: Combine dense and sparse results with configurable weighting (Reciprocal Rank Fusion).
- [ ] **Reranking Hooks**: Add a `rerank(callable)` interface compatible with Cross-Encoders and ColBERT.
- [ ] **Recall/Latency Benchmarking**: Tooling to help users tune HNSW/IVF parameters against an exact-search baseline.

---

## Technical Decision Rules
1. **Python-Native First**: If a feature can be implemented efficiently in NumPy/Python, do it there first to keep dependencies light.
2. **Portable Formats**: Favor Parquet and FAISS `.index` for interchange.
3. **API Ergonomics**: Prioritize a clean, discoverable API over internal complexity.
