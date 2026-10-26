import uuid

import pytest
from db_helpers import make_session, params, row

from database.mcps import MCP, MCPConfig, get_mcp_by_id, get_mcps

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
