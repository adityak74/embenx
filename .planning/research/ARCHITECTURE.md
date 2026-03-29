# Architecture Patterns (2025-2026)

**Project:** indexing_libs (Embenx)
**Researched:** 2025-05-22
**Overall confidence:** HIGH

## Recommended Architecture: Agentic-RAG Pipeline

Inspired by Uber's "Genie" (2025) and Pinterest's Agent Foundations (2026).

```mermaid
graph TD
    UserQuery[User/Agent Query] --> Optimizer[Query Optimizer Agent]
    Optimizer --> SubQueries[Multi-step Sub-queries]
    SubQueries --> Router[Dynamic Source Router]
    Router --> VectorIdx[Dense Vector Index (Qdrant/Faiss)]
    Router --> SparseIdx[Sparse Index (BM25/FTS5)]
    VectorIdx --> Merger[RRF Result Merger]
    SparseIdx --> Merger
    Merger --> PostProcessor[Post-Processor / Re-ranker]
    PostProcessor --> Context[Final Context for LLM]
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **Query Optimizer** | Breaking complex/ambiguous queries into smaller tasks. | Router |
| **Source Router** | Deciding which index (e.g., policy wiki vs. logs) to query. | Indexers |
| **Result Merger (RRF)** | Fusing results from diverse indexing strategies (current implementation). | Post-Processor |
| **MCP Adapter** | Exposing the entire pipeline as a tool for any MCP-compliant agent. | External Agents |

### Data Flow

1.  **Ingestion**: Documents are chunked, embedded, and indexed locally (Arrow/LanceDB).
2.  **Request**: An AI Agent issues a high-level goal (e.g., "Find all security policy violations in the 2026 logs").
3.  **Optimization**: The system creates a plan (sub-queries for "security policy" and "log search").
4.  **Retrieval**: Hybrid search returns ranked candidates.
5.  **Refinement**: If the agent finds the results insufficient, it issues a "trajectory pivot" (agentic loop).

## Patterns to Follow

### Pattern 1: Model Context Protocol (MCP) Integration
**What:** standard for exposing tools to LLMs.
**Why:** Agents in 2026 (OpenAI Deep Research, Anthropic) use MCP to discover and use retrieval tools.
**Example:**
```python
@mcp.tool()
def search_collection(query: str, collection_name: str):
    col = Collection.load(collection_name)
    return col.hybrid_search(query)
```

### Pattern 2: Matryoshka-Aware Retrieval
**What:** Using the first N dimensions of an embedding for coarse search, then full vector for re-ranking.
**Why:** Dramatically reduces initial search latency and RAM usage.

## Anti-Patterns to Avoid

### Anti-Pattern 1: "Naive Top-K Only"
**What:** Relying solely on semantic similarity without lexical (BM25) fallback.
**Why bad:** Misses exact keyword matches (IDs, specific terms) which are crucial for agents.
**Instead:** Always use Hybrid Search + RRF.

### Anti-Pattern 2: "Synchronous Retrieval Loops"
**What:** Blocking agent loops on long-running search tasks.
**Why bad:** Agents often need high-throughput parallel context gathering.
**Instead:** Use `async` retrieval and streaming result support.

## Scalability Considerations

| Concern | At 100 docs | At 10M docs | At 1B+ docs |
|---------|--------------|--------------|-------------|
| **Memory** | All-in-RAM (Faiss) | Hybrid RAM/Disk (Qdrant) | Disk-first (LanceDB/DiskANN) |
| **Latency** | <10ms | <50ms (optimized) | <200ms (with quantization) |
| **Concurrency** | Single-threaded | Multi-process/Async | Distributed (Milvus/Vespa) |

## Sources
- Uber Engineering Blog (May 2025) - "Genie" architecture.
- Pinterest (Mar 2026) - Agent foundations & MCP registry.
- OpenAI Deep Research (2025) - Trajectory-based retrieval.
