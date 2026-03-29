# Embenx Roadmap 🚀

This document outlines the strategic direction of Embenx as it evolves from a benchmarking CLI into a **Python-native embedding retrieval layer** and **Agentic World Memory toolkit**.

## Vision
Embenx aims to be the standard toolkit for Python developers to move embeddings and metadata through their pipelines, providing robust local retrieval, portable formats, and cognitive memory features for agentic world models.

---

## Milestone 1: Core Retrieval & Library API ✅
- [x] Unified Collection API, Library-Native Benchmarking, Refined I/O, and Quantization Support.

## Milestone 2: Metadata & Filtering ✅
- [x] Schema Model, Filtering DSL, and Parquet-Native Metadata.

## Milestone 3: Hybrid Search & Reranking ✅
- [x] Reranking Hooks, Sparse BM25 support, Hybrid Search (RRF), and Recall/Latency Benchmarking.

## Milestone 4: Advanced Reranking & Matryoshka (v0.4.0) 🚧
- [ ] **Matryoshka Truncation**: Native support for truncating MRL-capable embeddings.
- [ ] **Unified Rerank API**: Support for `sentence-transformers` and `rerankers`.
- [ ] **FlashRank Integration**: CPU-optimized reranking via ONNX.

## Milestone 5: Multi-modality & Agentic Memory (v0.5.0) 📅
- [ ] **Multimodal Collections**: Shared vector space for Text, Image, and Audio.
- [ ] **Stateful Conversational Memory**: Managed `Session` objects for long-term memory (LTM).
- [ ] **Time-Based Decay**: Rank results by similarity and recency.

## Milestone 6: World Models & Dynamic State (v0.6.0) 🌌
- [ ] **Trajectory Retrieval**: Search *sequences* of vectors (action/state trajectories).
- [ ] **Latent State Transitions**: Store and query transition matrices (State A -> Action -> State B).
- [ ] **Hippocampal Patterns**: Implement "Pattern Completion" (recalling whole trajectories from partial cues).

## Milestone 7: Retrieval-Augmented KV Caching (v0.7.0) ⚡
- [ ] **Activation Payloads**: Support for high-dimensional LLM activation tensors (via `safetensors`).
- [ ] **RA-KVC Engine**: Specialized collection for caching evicted KV pairs from vLLM/Llama.cpp.
- [ ] **RetrievalAttention (arXiv:2409.16148)**: Implement attention-aware vector indexing to retrieve <3% of KV tokens.

## Milestone 8: State Space Hydration (v0.8.0) 🌊
- [ ] **h0 Injection**: Prime initial hidden states ($h_0$) of SSMs (Mamba-2/Jamba).
- [ ] **Global Source of Truth**: External "Global State" for SSM agents to solve Exact Recall Amnesia.

## Milestone 9: Full Agentic MCP Integration (v0.9.0) 🤖
- [ ] **MCP Server Native**: Expose collections as Model Context Protocol (MCP) tools.
- [ ] **Autonomous Trajectories**: Support for autonomous multi-step search loops.

## Milestone 10: Research-Driven Optimizations (v1.0.0) 🧪
**Goal**: Integrate 2025-2026 SOTA algorithms from ArXiv.

- [ ] **ClusterKV (arXiv:2412.03213)**: Implement semantic clustering of KV/Vector pairs for 2x throughput.
- [ ] **TurboQuant (arXiv:2504.19874)**: Add 1-bit QJL and MSE-optimized quantization for activations.
- [ ] **ESWM (ICLR 2026)**: Neuroscience-inspired spatial cognitive maps for navigation trajectories.
- [ ] **Echo (arXiv:2502.16090)**: Temporal episodic memory logic for "what happened when" queries.

---

## Technical Decision Rules
1. **Python-Native First**: Implement efficiently in NumPy/Python first.
2. **Portable Formats**: Favor Parquet, FAISS `.index`, and `safetensors`.
3. **API Ergonomics**: Prioritize a clean, discoverable API.
