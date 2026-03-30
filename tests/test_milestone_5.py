import pytest
import asyncio
from unittest.mock import patch, MagicMock
from mcp_server import list_tools_impl, call_tool_impl

@pytest.mark.asyncio
async def test_mcp_list_tools():
    tools = await list_tools_impl()
    assert len(tools) == 3
    assert tools[0].name == "memory_add"
    assert tools[1].name == "memory_search"
    assert tools[2].name == "list_collections"

@pytest.mark.asyncio
async def test_mcp_call_list_collections():
    with patch("glob.glob", return_value=["test.parquet"]):
        res = await call_tool_impl("list_collections", {})
        assert "test" in res[0].text

@pytest.mark.asyncio
async def test_mcp_call_memory_add():
    with patch("mcp_server.Embedder") as mock_emb, \
         patch("mcp_server.Collection") as mock_col:
        mock_emb.return_value.embed_texts.return_value = [[0.1]*768]
        
        args = {
            "texts": ["hello world"],
            "collection": "test_mcp"
        }
        res = await call_tool_impl("memory_add", args)
        assert "Successfully added" in res[0].text
        mock_col.return_value.to_parquet.assert_called_once()

@pytest.mark.asyncio
async def test_mcp_call_memory_search():
    with patch("os.path.exists", return_value=True), \
         patch("core.Collection.from_parquet") as mock_from_pq, \
         patch("llm.Embedder") as mock_emb:
        
        mock_col = mock_from_pq.return_value
        mock_col.search.return_value = [({"text": "found it"}, 0.1)]
        mock_emb.return_value.embed_query.return_value = [0.1]*768
        
        args = {"query": "find me", "collection": "test_mcp"}
        res = await call_tool_impl("memory_search", args)
        assert "found it" in res[0].text

@pytest.mark.asyncio
async def test_mcp_call_memory_search_not_found():
    with patch("os.path.exists", return_value=False):
        args = {"query": "find me", "collection": "ghost"}
        res = await call_tool_impl("memory_search", args)
        assert "not found" in res[0].text

@pytest.mark.asyncio
async def test_mcp_call_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        await call_tool_impl("ghost_tool", {})
