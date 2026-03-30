import asyncio
import os
from typing import Any, Dict, List, Optional, Union

from mcp.server.stdio import stdio_server
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from core import Collection
from llm import Embedder

# --- MCP Server Implementation ---

app = Server("embenx-memory")

async def list_tools_impl() -> List[Tool]:
    """List available tools for agentic memory."""
    return [
        Tool(
            name="memory_add",
            description="Add embeddings and metadata to an Embenx collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "default": "default"},
                    "texts": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "array", "items": {"type": "object"}},
                    "model": {"type": "string", "default": "ollama/nomic-embed-text"}
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="memory_search",
            description="Search for relevant documents in an Embenx collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "default": "default"},
                    "query": {"type": "string"},
                    "top_k": {"type": "number", "default": 5},
                    "model": {"type": "string", "default": "ollama/nomic-embed-text"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_collections",
            description="List all local Embenx collections available.",
            inputSchema={"type": "object", "properties": {}}
        )
    ]

async def call_tool_impl(name: str, arguments: Dict[str, Any]) -> List[Union[TextContent, ImageContent, EmbeddedResource]]:
    """Handle tool calls from the AI agent."""
    if name == "memory_add":
        texts = arguments["texts"]
        meta = arguments.get("metadata")
        col_name = arguments.get("collection", "default")
        model = arguments.get("model", "ollama/nomic-embed-text")
        
        emb = Embedder(model)
        vectors = emb.embed_texts(texts)
        
        col = Collection(name=col_name, indexer_type="faiss")
        col.add(vectors, meta)
        path = f"{col_name}.parquet"
        col.to_parquet(path)
        
        return [TextContent(type="text", text=f"Successfully added {len(texts)} items to collection '{col_name}'. Storage: {path}")]

    elif name == "memory_search":
        query = arguments["query"]
        col_name = arguments.get("collection", "default")
        top_k = int(arguments.get("top_k", 5))
        model = arguments.get("model", "ollama/nomic-embed-text")
        
        path = f"{col_name}.parquet"
        if not os.path.exists(path):
            return [TextContent(type="text", text=f"Collection '{col_name}' not found. Path {path} does not exist.")]
            
        col = Collection.from_parquet(path)
        emb = Embedder(model)
        q_vec = emb.embed_query(query)
        
        results = col.search(q_vec, top_k=top_k)
        
        output = []
        for meta, dist in results:
            text = meta.get("text", str(meta))
            output.append(f"- {text} (Score: {dist:.4f})")
            
        return [TextContent(type="text", text="\n".join(output) if output else "No relevant matches found.")]

    elif name == "list_collections":
        import glob
        files = glob.glob("*.parquet")
        names = [f.replace(".parquet", "") for f in files]
        return [TextContent(type="text", text=f"Available collections: {', '.join(names) if names else 'None'}")]

    else:
        raise ValueError(f"Unknown tool: {name}")

# Register implementations with the app
app.list_tools()(list_tools_impl)
app.call_tool()(call_tool_impl)

async def run():
    async with stdio_server() as (read_stream, write_server):
        await app.run(read_stream, write_server, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(run())
