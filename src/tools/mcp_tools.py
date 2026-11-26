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


class MCPUnavailableError(Exception):
    """Raised when one or more MCP servers fail to load their tools."""

    def __init__(self, names: list[str]):
        self.names = names
        super().__init__(f"Unable to load tools from MCP server(s): {', '.join(names)}")


async def get_mcp_tools(
    session: AsyncSession,
    mcp_ids: list[uuid.UUID],
) -> list[BaseTool]:
    """Creates MCP tools for an AI agent to use.
    Raises MCPUnavailableError naming every MCP server that failed to load."""
    if not mcp_ids:
        return []
    # TODO: Replace with one function that can find all MCP ids in one turn.
    confs = await asyncio.gather(
        *[get_mcp_by_id(session=session, mcp_id=mcp_id) for mcp_id in mcp_ids]
    )
    conn: dict[str, Connection] = {
        conf.name: url_connection(url=f"{conf.url}/mcp", transport=conf.transport)
        for conf in confs
    }
    client = MultiServerMCPClient(connections=conn)
    log.debug(f"Created MultiServerMCPClient for {len(mcp_ids)} MCP(s)")

    # Load each server separately so a failure can be attributed to a specific MCP.
    names = list(conn)
    results = await asyncio.gather(
        *[client.get_tools(server_name=name) for name in names],
        return_exceptions=True,
    )
    tools: list[BaseTool] = []
    failed: list[str] = []
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            log.error(f"Failed to load tools from MCP '{name}'", exc_info=result)
            failed.append(name)
        elif isinstance(result, BaseException):
            raise result
        else:
            tools.extend(result)
    if failed:
        raise MCPUnavailableError(names=failed)
    return tools
