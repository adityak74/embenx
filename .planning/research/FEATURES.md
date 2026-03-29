# Feature Landscape

**Project:** indexing_libs (Embenx)
**Researched:** 2025-05-22
**Overall confidence:** HIGH

## Table Stakes (2025-2026)

Features essential for any vector search library to remain relevant in 2026.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Hybrid Search (Vector+BM25)** | Naive top-K similarity search is no longer sufficient. | Medium | Current implementation has RRF, which is excellent. |
| **Model Context Protocol (MCP)** | Agents need a standardized way to call retrieval tools. | Low/Medium | Essential for "agentic" interoperability. |
| **Metadata Filtering** | Business logic (time, tags, owners) must be combined with vectors. | Medium | Critical for real-world enterprise RAG. |
| **Quantization (Binary/PQ)** | To handle large indexes on consumer/edge hardware. | High | Necessary to break the "RAM wall" (USearch/Faiss/Qdrant support this). |

## Differentiators

Features requested by developers in early 2026 discussions.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Agentic Trajectories** | Multi-step search that allows the system to backtrack or pivot. | High | Inspired by OpenAI's Deep Research. |
| **Generative Retrieval** | Predicting document IDs directly instead of vector search. | High | Pinterest-style approach for 2026 scalability. |
| **Matryoshka Embeddings (MRL)** | Efficient variable-precision search (coarse -> fine). | Medium | Allows speed/accuracy tradeoff. |
| **Sync-Ready Persistence** | Local-first sync with cloud counterparts (PowerSync style). | Medium/High | Enabling "offline-first" AI apps. |
| **Dynamic Source Identifier** | Agent autonomously choosing the right index to query. | Medium | Uber "Genie" approach for heterogeneous data. |

## Anti-Features

Features to explicitly NOT build (or de-prioritize).

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Cloud Hosting | Expensive to maintain; violates "local-first" philosophy. | Focus on portable local formats (Lance/Parquet). |
| Custom Model Training | Too much overhead; models are commoditized. | Focus on "flexible re-indexing" and migration tools. |
| Visual Dashboard | Bloats the library; developer-first toolkits are preferred. | Provide clean APIs and structured exports (JSON/CSV). |

## Feature Dependencies

```
Hybrid Search → Model Context Protocol (MCP) → Agentic Trajectories
(MCP makes it easier for agents to handle the complexity of hybrid search results)
```

## MVP Recommendation

Prioritize:
1. **MCP Tool Implementation**: Enable `Embenx` collections to be exposed as MCP tools for AI agents.
2. **Metadata Filtering (Advanced)**: More robust SQL-like or Pydantic-based filtering across indexers.
3. **LanceDB Integration**: Support for Arrow-backed persistent storage for large local datasets.

Defer: **Generative Retrieval**: High complexity, better suited for massive web-scale implementations.

## Sources
- Reddit (r/Rag) - Discussions on "Re-indexing nightmare" and model lock-in.
- X (Twitter) - Feed on "AI Agents breaking the query model."
- Pinterest Engineering Blog - Transition to MCP for Agent foundations.
- Uber Engineering Blog - Multi-agent retrieval pipelines.
