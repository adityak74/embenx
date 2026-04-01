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
- [x] **Agentic Memory loops**: Integrated feedback loops for self-healing ranking.
- [x] **Temporal Memory**: Managed `Session` objects with time-based decay for long-term agentic context.

## Milestone 6: Visualizer & DevTools ✅
**Goal**: Provide the best developer experience in the ecosystem.

- [x] **Embenx Explorer**: A lightweight, built-in web UI to visualize vector clusters and metadata distributions (built with Streamlit).
- [x] **HNSW Graph Visualizer**: Interactive 3D visualization of the HNSW graph layers and navigation paths.
- [x] **RAG Playground**: Integrated chat loop to test retrieval quality with LLMs.
- [x] **Export to production**: One-click export from local Embenx collections to production clusters (Qdrant, Milvus).

## Milestone 7: Advanced World Models ✅
- [x] **Trajectory Retrieval**: Search *sequences* of vectors (action/state trajectories) with mean/max pooling.
- [x] **Retrieval-Augmented KV Caching (RA-KVC)**: Off-path KV storage using `safetensors`.
- [x] **State Space Hydration**: Prime initial hidden states ($h_0$) of SSMs (Mamba-2) using `StateCollection`.

## Milestone 8: Research-Driven Optimizations ✅
**Goal**: Integrate 2025-2026 SOTA algorithms from ArXiv.

- [x] **ClusterKV (arXiv:2412.03213)**: Implement semantic clustering of KV/Vector pairs for improved throughput.
- [x] **TurboQuant (arXiv:2504.19874)**: Add 1-bit sign-based quantization for aggressive activation compression.
- [x] **ESWM (ICLR 2026)**: Neuroscience-inspired spatial cognitive maps for navigation trajectories.
- [x] **Echo (arXiv:2502.16090)**: Temporal episodic memory logic for "what happened when" queries.

## Milestone 9: Full Agentic MCP Integration ✅
**Goal**: Universal interoperability for the Agentic Era.
- [x] **MCP Server Native**: Expose Embenx collections as Model Context Protocol (MCP) tools.
- [x] **Agentic Trajectories**: Support for autonomous query refinement using `AgenticCollection`.
- [x] **Self-Healing Retrieval**: Integrated feedback loops to automatically update rankings.

## Milestone 10: Final Polish & Ecosystem Launch ✅
**Goal**: The gold standard for retrieval engineering.
- [x] **Retrieval Zoo**: Launched pre-built Embenx collections via `embenx zoo-load`.
- [x] **Technical Report Generator**: Automated generation of Markdown benchmark reports.
- [x] **Multimodal Support**: Native handling of image embeddings (CLIP).
- [x] **100% Documentation Coverage**: Every class, method, and script fully documented.

---

## Strategic Growth Decision Rules (The Path to 5,000 Stars)
1. **Local-First, Cloud-Optional**: Prioritize the developer's laptop. Zero-setup is our superpower.
2. **Interoperability is King**: Support every major format (Arrow, Parquet, FAISS, Safetensors) and protocol (MCP).
3. **Show, Don't Just Tell**: Invest heavily in the Visualizer (Milestone 6).
4. **Performance for Free**: Use Rust-backed libs and ONNX to ensure Embenx is always the fastest tool.
