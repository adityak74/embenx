# Embenx Roadmap 🚀

This document outlines the strategic direction of Embenx as it evolves from a benchmarking CLI into a **Universal Embedding Retrieval Toolkit** and **Agentic Memory Layer**.

## Vision
Embenx aims to be the standard toolkit for Python developers to move embeddings and metadata through their pipelines, providing robust local retrieval, portable formats, and high-performance memory for agentic workflows.

---

## Milestone 1: Core Retrieval & Library API ✅
- [x] Unified Collection API, Library-Native Benchmarking, Refined I/O, and Quantization Support.

## Milestone 2: Metadata & Filtering ✅
- [x] Schema Model, Filtering DSL, and Parquet-Native Metadata.

## Milestone 3: Hybrid Search & Reranking ✅
- [x] Reranking Hooks, Sparse BM25 support, Hybrid Search (RRF), and Recall/Latency Benchmarking.

## Milestone 4: Advanced Reranking & Matryoshka (v0.4.0) ✅
**Goal**: Implement "Shortlist & Rerank" patterns for 2026-grade performance.

- [x] **Matryoshka Truncation**: Native support for truncating MRL-capable embeddings.
- [x] **Unified Rerank API**: Integrated support for `sentence-transformers` and `rerankers` library.
- [x] **FlashRank Integration**: CPU-optimized reranking via ONNX for sub-100ms latency.
- [x] **Late Interaction (ColBERT)**: Supported via `rerankers` integration.

## Milestone 5: Agentic Era & MCP (v0.5.0) ✅
**Goal**: Become the default memory for AI Agents (Claude, GPT-5).

- [x] **MCP Server Native**: Expose Embenx collections as Model Context Protocol (MCP) tools for instant integration with Claude Desktop and other agentic IDEs.
- [ ] **Agentic Trajectories**: Support for autonomous multi-step search loops where the model refines its own queries.
- [ ] **Temporal Memory**: Managed `Session` objects with time-based decay for long-term agentic context.

## Milestone 6: Visualizer & DevTools (v0.6.0) 🚧
**Goal**: Provide the best developer experience in the ecosystem.

- [ ] **Embenx Explorer**: A lightweight, built-in web UI to visualize vector clusters and metadata distributions (built with Streamlit or FastAPI/React).
- [ ] **HNSW Graph Visualizer**: Interactive 3D visualization of the HNSW graph traversal during search.
- [ ] **Export to production**: One-click export from local Embenx collections to production clusters.

## Milestone 7: Advanced World Models (v0.7.0) 🌌
- [ ] **Trajectory Retrieval**: Search *sequences* of vectors (action/state trajectories).
- [ ] **Retrieval-Augmented KV Caching (RA-KVC)**: Off-path KV storage to break the context-window "RAM Wall."
- [ ] **State Space Hydration**: Prime initial hidden states ($h_0$) of SSMs (Mamba-2).

---

## Strategic Growth Decision Rules (The Path to 5,000 Stars)
1. **Local-First, Cloud-Optional**: Prioritize the developer's laptop. Zero-setup is our superpower.
2. **Interoperability is King**: Support every major format (Arrow, Parquet, FAISS, Safetensors) and protocol (MCP).
3. **Show, Don't Just Tell**: Invest heavily in the Visualizer (Milestone 6).
4. **Performance for Free**: Use Rust-backed libs and ONNX to ensure Embenx is always the fastest tool.
