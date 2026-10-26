from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from core.config import config
from core.logging import get_logger
from tools.mcp_tools import url_connection

log = get_logger()


async def get_web_search_tool() -> list[BaseTool]:
    """Returns Firecrawl web search tools from the configured MCP server."""
    # Self hosted firecrawl API does not align with self hosted MCP.
    # Filter to only firecrawl_map, firecrawl_scrape, and firecrawl_crawl.
    conn = {
        "Web Search": url_connection(
            url=config.WEB_SEARCH_URL, transport=config.WEB_SEARCH_TRANSPORT
        )
    }
    client = MultiServerMCPClient(connections=conn)
    log.debug("Created MultiServerMCPClient for Web Search")
    tools = await client.get_tools()
    allowed_firecrawl_tools = {"firecrawl_map", "firecrawl_scrape", "firecrawl_crawl"}
    return [tool for tool in tools if tool.name in allowed_firecrawl_tools]
