# Embenx Roadmap 🚀

This document outlines the strategic direction of Embenx as it evolves from a benchmarking CLI into a **Python-native embedding retrieval layer** and **Agentic World Memory toolkit**.

## Vision
Embenx aims to be the standard toolkit for Python developers to move embeddings and metadata through their pipelines, providing robust local retrieval, portable formats, and cognitive memory features for agentic world models.

---

## Milestone 1: Core Retrieval & Library API ✅
**Goal**: Transition from CLI-only to a usable Python library with a stable API.

- [x] **Unified Collection API**: Implement a `Collection` class to wrap indexers.
- [x] **Library-Native Benchmarking**: Compare performance directly from the Collection API.
- [x] **Refined I/O**: Native Parquet, NumPy (.npy/.npz), and FAISS (.index) loading.
- [x] **Quantization Support**: FAISS SQ8/PQ and USearch F16/I8.

## Milestone 2: Metadata & Filtering ✅
**Goal**: Make retrieval production-aware with schemas and filters.

- [x] **Schema Model**: Internal storage of `id`, `vector`, `text`, and `metadata`.
- [x] **Filtering DSL**: Implement a `where` clause for metadata filtering.
- [x] **Parquet-Native Schema**: Ensure metadata travels with vectors in Parquet exports.

## Milestone 3: Hybrid Search & Reranking ✅
**Goal**: Achieve state-of-the-art retrieval quality.

- [x] **Reranking Hooks**: Add a `rerank(callable)` interface.
- [x] **Sparse Vector Support**: Added BM25Indexer.
- [x] **Hybrid Search**: Combined dense and sparse results with RRF.
- [x] **Recall/Latency Benchmarking**: Tooling to tune ANN indexers against exact search.

## Milestone 4: Advanced Reranking & Matryoshka (v0.4.0) 🚧
**Goal**: Implement "Shortlist & Rerank" patterns for 2026-grade performance.

- [ ] **Matryoshka Truncation**: Native support for truncating MRL-capable embeddings for ultra-fast first-stage retrieval.
- [ ] **Unified Rerank API**: Integrated support for `sentence-transformers` and `rerankers` library.
- [ ] **FlashRank Integration**: Ultra-fast CPU-optimized reranking via ONNX.

## Milestone 5: Multi-modality & Agentic Memory (v0.5.0) 📅
**Goal**: Transition to multimodal retrieval and stateful memory for AI agents.

- [ ] **Multimodal Collections**: Shared vector space support for Text, Image, and Audio.
- [ ] **Stateful Conversational Memory**: Managed `Session` objects for long-term memory (LTM).
- [ ] **Time-Based Decay**: Rank results by both similarity and temporal relevance (recency bias).

## Milestone 6: World Models & Dynamic State (v0.6.0) 🌌
**Goal**: Support latent state management for World Models (JEPA, Gato).

- [ ] **Trajectory Retrieval**: Support for searching *sequences* of vectors (action/state trajectories) rather than single points.
- [ ] **Latent State Transitions**: Store and query transition matrices (State A -> Action -> State B) for predictive retrieval.
- [ ] **Grounding Hooks**: Tools to compare "simulated" latent states from world models against "real" retrieved episodic memories.
- [ ] **Hippocampal Patterns**: Implement "Pattern Completion" (recalling whole trajectories from partial cues).

---

## Technical Decision Rules
1. **Python-Native First**: Implement efficiently in NumPy/Python first to keep dependencies light.
2. **Portable Formats**: Favor Parquet and FAISS `.index` for interchange.
3. **API Ergonomics**: Prioritize a clean, discoverable API over internal complexity.
