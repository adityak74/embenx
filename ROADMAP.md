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

## Milestone 4: Advanced Reranking & Matryoshka ✅
**Goal**: Implement "Shortlist & Rerank" patterns for 2026-grade performance.

- [x] **Matryoshka Truncation**: Native support for truncating MRL-capable embeddings.
- [x] **Unified Rerank API**: Integrated support for `sentence-transformers` and `rerankers` library.
- [x] **FlashRank Integration**: CPU-optimized reranking via ONNX for sub-100ms latency.
- [x] **Late Interaction (ColBERT)**: Supported via `rerankers` integration.

## Milestone 5: Agentic Era & MCP ✅
**Goal**: Become the default memory for AI Agents (Claude, GPT-5).

- [x] **MCP Server Native**: Expose Embenx collections as Model Context Protocol (MCP) tools for instant integration with Claude Desktop and other agentic IDEs.
- [ ] **Agentic Trajectories**: Support for autonomous multi-step search loops where the model refines its own queries.
- [ ] **Temporal Memory**: Managed `Session` objects with time-based decay for long-term agentic context.

## Milestone 6: Visualizer & DevTools ✅
**Goal**: Provide the best developer experience in the ecosystem.

- [x] **Embenx Explorer**: A lightweight, built-in web UI to visualize vector clusters and metadata distributions (built with Streamlit).
- [ ] **HNSW Graph Visualizer**: Interactive 3D visualization of the HNSW graph traversal during search.
- [ ] **Export to production**: One-click export from local Embenx collections to production clusters.

## Milestone 7: Advanced World Models ✅
- [x] **Trajectory Retrieval**: Search *sequences* of vectors (action/state trajectories) with mean/max pooling.
- [x] **Retrieval-Augmented KV Caching (RA-KVC)**: Off-path KV storage using `safetensors`.
- [x] **State Space Hydration**: Prime initial hidden states ($h_0$) of SSMs (Mamba-2) using `StateCollection`.

## Milestone 8: Research-Driven Optimizations (v1.0.0) ✅
**Goal**: Integrate 2025-2026 SOTA algorithms from ArXiv.

- [x] **ClusterKV (arXiv:2412.03213)**: Implement semantic clustering of KV/Vector pairs for improved throughput.
- [x] **TurboQuant (arXiv:2504.19874)**: Add 1-bit sign-based quantization for aggressive activation compression.
- [x] **ESWM (ICLR 2026)**: Neuroscience-inspired spatial cognitive maps for navigation trajectories.
- [x] **Echo (arXiv:2502.16090)**: Temporal episodic memory logic for "what happened when" queries.

---

## Strategic Growth Decision Rules (The Path to 5,000 Stars)
1. **Local-First, Cloud-Optional**: Prioritize the developer's laptop. Zero-setup is our superpower.
2. **Interoperability is King**: Support every major format (Arrow, Parquet, FAISS, Safetensors) and protocol (MCP).
3. **Show, Don't Just Tell**: Invest heavily in the Visualizer (Milestone 6).
4. **Performance for Free**: Use Rust-backed libs and ONNX to ensure Embenx is always the fastest tool.
