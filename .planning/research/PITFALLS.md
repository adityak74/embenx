# Domain Pitfalls (2025-2026)

**Project:** indexing_libs (Embenx)
**Researched:** 2025-05-22
**Overall confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Embedding Model Lock-in (Re-indexing Nightmare)
**What goes wrong:** Upgrading to a 2026 SOTA embedding model requires re-embedding and re-indexing the entire dataset.
**Why it happens:** Vectors are model-specific; a distance of 0.5 in `text-embedding-3` means something different in `2026-nova-base`.
**Consequences:** Massive compute cost and potential downtime for billions of vectors.
**Prevention:** 
1. Use **Matryoshka Embeddings** where possible to allow dimensionality changes without full re-indexing. 
2. Maintain a "Canonical Source" (Arrow/Parquet) to make re-indexing as fast as possible.
3. Use semantic caching for frequent queries to mitigate the cost during migration.

### Pitfall 2: The "RAM Wall"
**What goes wrong:** Using HNSW (Hierarchical Navigable Small World) indexes for 10M+ vectors exceeds available RAM.
**Why it happens:** HNSW keeps the graph structure in memory for speed.
**Consequences:** OOM (Out of Memory) crashes or extreme cloud costs for high-RAM instances.
**Prevention:** Use disk-based indexing (DiskANN, LanceDB) or quantization (Product/Binary) for large local datasets.

## Moderate Pitfalls

### Pitfall 3: Agentic "Query Pressure"
**What goes wrong:** AI Agents issue hundreds of parallel queries, overwhelming the single-threaded retrieval logic.
**Why it happens:** Traditional RAG was built for human-paced queries (1/sec); agents operate at 100/sec.
**Consequences:** High latency, timeout errors in agent loops.
**Prevention:** Implement `async` search and use high-concurrency optimized backends like Qdrant.

### Pitfall 4: Naive "Top-K" Precision Gap
**What goes wrong:** Search returns semantically similar "slop" that lacks specific keywords.
**Why it happens:** Pure vector search doesn't respect exact matches (e.g., "Error Code 404").
**Consequences:** LLM hallucinates because the retrieved context is "close but wrong."
**Prevention:** **Mandatory Hybrid Search (Vector + BM25)**.

## Minor Pitfalls

### Pitfall 5: Fragmentation of Data Silos
**What goes wrong:** Storing metadata in SQL and vectors in a specialized DB leads to complex "join" logic.
**Prevention:** Use consolidated formats (PostgreSQL with `pgvector` or local LanceDB files).

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Ingestion | Large file bottleneck | Use Arrow/Parquet for zero-copy data loading. |
| Retrieval | Stale context | Implement real-time index updates for "agentic memory." |
| Agent Integration | Non-standard tool calling | Adopt **Model Context Protocol (MCP)** immediately. |

## Sources
- Reddit (r/Rag) - Discussions on 2025-2026 operational bottlenecks.
- X (Twitter) - Developer complaints about specialized vector DB complexity.
- Pinterest Engineering Blog - Lessons from scaling to 300B+ Pins.
