from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.web_search import get_web_search_tool

pytestmark = pytest.mark.asyncio


class TestGetWebSearchTool:
    async def test_returns_filtered_firecrawl_tools(self):
        """Test that only allowed Firecrawl tools are returned."""
        mock_firecrawl_map = MagicMock()
        mock_firecrawl_map.name = "firecrawl_map"
        mock_firecrawl_scrape = MagicMock()
        mock_firecrawl_scrape.name = "firecrawl_scrape"
        mock_firecrawl_crawl = MagicMock()
        mock_firecrawl_crawl.name = "firecrawl_crawl"
        mock_other_tool = MagicMock()
        mock_other_tool.name = "some_other_tool"

        with patch("tools.web_search.MultiServerMCPClient") as mock_client_class:
            with patch("tools.web_search.config") as mock_config:
                mock_config.WEB_SEARCH_URL = "http://web-search"
                mock_config.WEB_SEARCH_TRANSPORT = "sse"

                mock_client = MagicMock()
                mock_client.get_tools = AsyncMock(
                    return_value=[
                        mock_firecrawl_map,
                        mock_firecrawl_scrape,
                        mock_firecrawl_crawl,
                        mock_other_tool,
                    ]
                )
                mock_client_class.return_value = mock_client

                result = await get_web_search_tool()

                assert mock_firecrawl_map in result
                assert mock_firecrawl_scrape in result
                assert mock_firecrawl_crawl in result
                assert mock_other_tool not in result
                assert len(result) == 3

    async def test_configures_client_with_web_search_url(self):
        """Test that client is configured with the correct web search URL and transport."""
        with patch("tools.web_search.MultiServerMCPClient") as mock_client_class:
            with patch("tools.web_search.config") as mock_config:
                mock_config.WEB_SEARCH_URL = "http://custom-search"
                mock_config.WEB_SEARCH_TRANSPORT = "custom-transport"

                mock_client = MagicMock()
                mock_client.get_tools = AsyncMock(return_value=[])
                mock_client_class.return_value = mock_client

                await get_web_search_tool()

                mock_client_class.assert_called_once_with(
                    connections={
                        "Web Search": {
                            "url": "http://custom-search",
                            "transport": "custom-transport",
                        }
                    }
                )

    async def test_returns_empty_list_when_no_tools_available(self):
        """Test that empty list is returned when no tools are available."""
        with patch("tools.web_search.MultiServerMCPClient") as mock_client_class:
            with patch("tools.web_search.config") as mock_config:
                mock_config.WEB_SEARCH_URL = "http://web-search"
                mock_config.WEB_SEARCH_TRANSPORT = "sse"

                mock_client = MagicMock()
                mock_client.get_tools = AsyncMock(return_value=[])
                mock_client_class.return_value = mock_client

                result = await get_web_search_tool()

                assert result == []

    async def test_preserves_tool_order(self):
        """Test that tool filtering preserves the order of allowed tools."""
        mock_firecrawl_map = MagicMock()
        mock_firecrawl_map.name = "firecrawl_map"
        mock_firecrawl_scrape = MagicMock()
        mock_firecrawl_scrape.name = "firecrawl_scrape"
        mock_firecrawl_crawl = MagicMock()
        mock_firecrawl_crawl.name = "firecrawl_crawl"

        with patch("tools.web_search.MultiServerMCPClient") as mock_client_class:
            with patch("tools.web_search.config") as mock_config:
                mock_config.WEB_SEARCH_URL = "http://web-search"
                mock_config.WEB_SEARCH_TRANSPORT = "sse"

                mock_client = MagicMock()
                mock_client.get_tools = AsyncMock(
                    return_value=[
                        mock_firecrawl_map,
                        mock_firecrawl_scrape,
                        mock_firecrawl_crawl,
                    ]
                )
                mock_client_class.return_value = mock_client

                result = await get_web_search_tool()

                assert result == [
                    mock_firecrawl_map,
                    mock_firecrawl_scrape,
                    mock_firecrawl_crawl,
                ]
