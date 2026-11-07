import uuid

import pytest
from db_helpers import make_session, params, row

from database.mcps import MCP, MCPConfig, get_mcp_by_id, get_mcps, insert_mcp

pytestmark = pytest.mark.asyncio


class TestGetMcps:
    async def test_returns_all_mcps(self):
        a, b = uuid.uuid4(), uuid.uuid4()
        session = make_session(
            scalars=[[row(id=a, name="search"), row(id=b, name="files")]]
        )

        assert await get_mcps(session=session) == [
            MCP(id=a, name="search"),
            MCP(id=b, name="files"),
        ]

    async def test_returns_empty_list_when_none(self):
        session = make_session(scalars=[[]])

        assert await get_mcps(session=session) == []


class TestGetMcpById:
    async def test_returns_config(self):
        mcp_id = uuid.uuid4()
        session = make_session(
            execute=[
                row(
                    id=mcp_id,
                    name="search",
                    url="http://mcp",
                    transport="streamable_http",
                )
            ]
        )

        result = await get_mcp_by_id(session=session, mcp_id=mcp_id)

        assert result == MCPConfig(
            id=mcp_id, name="search", url="http://mcp", transport="streamable_http"
        )
        assert params(session.execute.await_args) == [mcp_id]

    async def test_raises_when_not_found(self):
        session = make_session(execute=[None])

        with pytest.raises(ValueError, match="does not exist"):
            await get_mcp_by_id(session=session, mcp_id=uuid.uuid4())


class TestInsertMcp:
    async def test_returns_id(self):
        session = make_session()
        config = MCPConfig(
            name="test_mcp",
            url="http://localhost:8000",
            transport="streamable_http",
        )

        result = await insert_mcp(session=session, mcp_config=config)

        assert isinstance(result, uuid.UUID)

    async def test_adds_mcp_with_correct_fields(self):
        session = make_session()
        config = MCPConfig(
            name="search",
            url="http://mcp-server",
            transport="sse",
        )

        await insert_mcp(session=session, mcp_config=config)

        assert len(session.added) == 1
        added = session.added[0]
        assert added.name == "search"
        assert added.url == "http://mcp-server"
        assert added.transport == "sse"
        assert isinstance(added.id, uuid.UUID)

    async def test_uses_provided_id(self):
        session = make_session()
        provided_id = uuid.uuid4()
        config = MCPConfig(
            id=provided_id,
            name="test",
            url="http://example.com",
            transport="http",
        )

        result = await insert_mcp(session=session, mcp_config=config)

        assert result == provided_id
        assert session.added[0].id == provided_id
