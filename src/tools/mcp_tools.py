import asyncio
import uuid

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    Connection,
    SSEConnection,
    StreamableHttpConnection,
    WebsocketConnection,
)
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.mcps import get_mcp_by_id

log = get_logger()


def url_connection(url: str, transport: str) -> Connection:
    """Builds a URL based MCP connection.
    Raises ValueError if the transport is not supported for URL based connections."""
    match transport:
        case "sse":
            return SSEConnection(transport="sse", url=url)
        case "streamable_http" | "streamable-http" | "http":
            return StreamableHttpConnection(transport="streamable_http", url=url)
        case "websocket":
            return WebsocketConnection(transport="websocket", url=url)
    raise ValueError(f"Unsupported MCP transport '{transport}' for url '{url}'")


async def get_mcp_tools(
    session: AsyncSession,
    mcp_ids: list[uuid.UUID],
) -> list[BaseTool]:
    """Creates MCP tools for an AI agent to use."""
    # TODO: Replace with one function that can find all MCP ids in one turn.
    conn: dict[str, Connection] = {}
    if mcp_ids:
        confs = await asyncio.gather(
            *[get_mcp_by_id(session=session, mcp_id=mcp_id) for mcp_id in mcp_ids]
        )
        for conf in confs:
            conn[conf.name] = url_connection(
                url=f"{conf.url}/mcp", transport=conf.transport
            )
    client = MultiServerMCPClient(connections=conn)
    log.debug(f"Created MultiServerMCPClient for {len(mcp_ids)} MCP(s)")
    return await client.get_tools()
