# Technology Stack

**Project:** indexing_libs (Embenx)
**Researched:** 2025-05-22
**Overall confidence:** HIGH

## Recommended Stack for 2025-2026

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.12+ | Core Logic | Modern features like advanced typing and performance improvements. |
| PyArrow | 15.0+ | Data Exchange | Standard for local-first, zero-copy data transport between indexers and memory. |
| Model Context Protocol (MCP) | 1.8+ | Agent Tooling | Industry standard (OpenAI, Anthropic, Pinterest) for connecting agents to retrieval tools. |

### Database / Indexing (Local-First Focus)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| LanceDB | 0.6+ | Persistent Storage | Built on Arrow; delivers near in-memory speed from disk. Handles "RAM wall" bottleneck. |
| Qdrant | 1.17+ | Edge High-Perf | Best-in-class Rust implementation for high-concurrency agentic tasks. |
| SQLite + FTS5 | 3.45+ | Sparse Indexing | Lightweight, local-first alternative to BM25 for metadata and keyword search. |

### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rank-bm25 | 0.2.2 | BM25 Sparse | For quick keyword-based retrieval in hybrid search. |
| usearch | 2.23+ | Vector Engine | Extremely lightweight (single header) ANN for edge devices. |
| pydantic | 2.6+ | Schema Validation | Type-safe metadata handling for agentic tool use. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Storage | LanceDB | Pinecone | Pinecone is cloud-only; violates "local-first" requirement. |
| Vector Engine | Qdrant | Faiss | Faiss lacks built-in persistence and metadata filtering required by agents. |
| Retrieval | MCP | Custom JSON-RPC | MCP is becoming the industry standard (OpenAI/Pinterest); better interoperability. |

## Installation

```bash
# Core Dependencies
pip install pyarrow lancedb qdrant-client rank-bm25 mcp

# Optional for specialized indexers
pip install usearch hnswlib
```

## Sources
- Pinterest Engineering Blog (2025-2026) - Agent Foundations & MCP
- Uber Engineering Blog (May 2025) - Genie Agentic-RAG
- OpenAI (2025) - Deep Research & o3 model findings
- Reddit (r/Rag, r/vectordatabase) 2025-2026 trends
