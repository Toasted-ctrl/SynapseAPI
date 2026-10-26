import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.get_tools import get_agent_tools

pytestmark = pytest.mark.asyncio


class TestGetAgentTools:
    async def test_returns_all_enabled_tools(self):
        """Test that all enabled tools are returned."""
        user_id = uuid.uuid4()
        mcp_ids = [uuid.uuid4()]

        mock_session = AsyncMock()
        mock_pr = MagicMock()

        mock_mcp_tool = MagicMock()
        mock_web_tool = MagicMock()
        mock_vs_files_tool = MagicMock()
        mock_vs_memories_tool = MagicMock()
        mock_skill_tool = MagicMock()

        with patch(
            "tools.get_tools.get_mcp_tools", new_callable=AsyncMock
        ) as mock_get_mcp:
            with patch(
                "tools.get_tools.get_web_search_tool", new_callable=AsyncMock
            ) as mock_get_web:
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ) as mock_get_vs:
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ) as mock_get_skill:
                        mock_get_mcp.return_value = [mock_mcp_tool]
                        mock_get_web.return_value = [mock_web_tool]
                        mock_get_vs.side_effect = [
                            mock_vs_files_tool,
                            mock_vs_memories_tool,
                        ]
                        mock_get_skill.return_value = mock_skill_tool

                        result = await get_agent_tools(
                            mcp_ids=mcp_ids,
                            session=mock_session,
                            web_search=True,
                            user_vs_files=True,
                            user_vs_memories=True,
                            user_vs_skills=True,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        assert len(result) == 5
                        assert mock_mcp_tool in result
                        assert mock_web_tool in result
                        assert mock_vs_files_tool in result
                        assert mock_vs_memories_tool in result
                        assert mock_skill_tool in result

    async def test_skips_disabled_web_search(self):
        """Test that web search tool is not included when disabled."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()

        with patch(
            "tools.get_tools.get_mcp_tools", new_callable=AsyncMock
        ) as mock_get_mcp:
            with patch(
                "tools.get_tools.get_web_search_tool", new_callable=AsyncMock
            ) as mock_get_web:
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ) as mock_get_vs:
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ):
                        mock_get_mcp.return_value = []
                        mock_get_web.return_value = []
                        mock_get_vs.return_value = MagicMock()

                        result = await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=False,
                            user_vs_files=True,
                            user_vs_memories=False,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        mock_get_web.assert_not_called()
                        assert len(result) == 1

    async def test_skips_disabled_vs_tools(self):
        """Test that vs tools are not included when disabled."""
        user_id = uuid.uuid4()
        mcp_ids = [uuid.uuid4()]
        mock_session = AsyncMock()
        mock_pr = MagicMock()

        mock_mcp_tool = MagicMock()
        mock_web_tool = MagicMock()

        with patch(
            "tools.get_tools.get_mcp_tools", new_callable=AsyncMock
        ) as mock_get_mcp:
            with patch(
                "tools.get_tools.get_web_search_tool", new_callable=AsyncMock
            ) as mock_get_web:
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ) as mock_get_vs:
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ):
                        mock_get_mcp.return_value = [mock_mcp_tool]
                        mock_get_web.return_value = [mock_web_tool]

                        result = await get_agent_tools(
                            mcp_ids=mcp_ids,
                            session=mock_session,
                            web_search=True,
                            user_vs_files=False,
                            user_vs_memories=False,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        mock_get_vs.assert_not_called()
                        assert len(result) == 2

    async def test_skips_mcp_tools_when_no_ids(self):
        """Test that mcp tools are not fetched when no IDs are provided."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()

        with patch(
            "tools.get_tools.get_mcp_tools", new_callable=AsyncMock
        ) as mock_get_mcp:
            with patch(
                "tools.get_tools.get_web_search_tool", new_callable=AsyncMock
            ) as mock_get_web:
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ):
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ):
                        mock_get_web.return_value = []

                        await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=True,
                            user_vs_files=False,
                            user_vs_memories=False,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        mock_get_mcp.assert_not_called()

    async def test_returns_empty_list_when_all_disabled(self):
        """Test that empty list is returned when all tools are disabled."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()

        with patch("tools.get_tools.get_mcp_tools", new_callable=AsyncMock):
            with patch("tools.get_tools.get_web_search_tool", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ):
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ):
                        result = await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=False,
                            user_vs_files=False,
                            user_vs_memories=False,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        assert result == []

    async def test_passes_correct_parameters_to_mcp_tools(self):
        """Test that correct parameters are passed to get_mcp_tools."""
        user_id = uuid.uuid4()
        mcp_ids = [uuid.uuid4(), uuid.uuid4()]
        mock_session = AsyncMock()
        mock_pr = MagicMock()

        with patch(
            "tools.get_tools.get_mcp_tools", new_callable=AsyncMock
        ) as mock_get_mcp:
            with patch("tools.get_tools.get_web_search_tool", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ):
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ):
                        mock_get_mcp.return_value = []

                        await get_agent_tools(
                            mcp_ids=mcp_ids,
                            session=mock_session,
                            web_search=False,
                            user_vs_files=False,
                            user_vs_memories=False,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        mock_get_mcp.assert_called_once_with(
                            session=mock_session, mcp_ids=mcp_ids
                        )

    async def test_passes_correct_parameters_to_vs_search_tools(self):
        """Test that correct parameters are passed to get_vs_search_tool."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()

        with patch("tools.get_tools.get_mcp_tools", new_callable=AsyncMock):
            with patch("tools.get_tools.get_web_search_tool", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ) as mock_get_vs:
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ):
                        mock_get_vs.return_value = MagicMock()

                        await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=False,
                            user_vs_files=True,
                            user_vs_memories=True,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        assert mock_get_vs.call_count == 2
                        calls = mock_get_vs.call_args_list

                        assert calls[0].kwargs == {
                            "session": mock_session,
                            "user_id": user_id,
                            "scope": "user_vs_files",
                            "pr": mock_pr,
                        }
                        assert calls[1].kwargs == {
                            "session": mock_session,
                            "user_id": user_id,
                            "scope": "user_vs_memories",
                            "pr": mock_pr,
                        }

    async def test_only_user_vs_files_enabled(self):
        """Test when only user_vs_files is enabled."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()
        mock_vs_tool = MagicMock()

        with patch("tools.get_tools.get_mcp_tools", new_callable=AsyncMock):
            with patch("tools.get_tools.get_web_search_tool", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ) as mock_get_vs:
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ):
                        mock_get_vs.return_value = mock_vs_tool

                        result = await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=False,
                            user_vs_files=True,
                            user_vs_memories=False,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        mock_get_vs.assert_called_once_with(
                            session=mock_session,
                            user_id=user_id,
                            scope="user_vs_files",
                            pr=mock_pr,
                        )
                        assert result == [mock_vs_tool]

    async def test_only_user_vs_memories_enabled(self):
        """Test when only user_vs_memories is enabled."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()
        mock_vs_tool = MagicMock()

        with patch(
            "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
        ) as mock_get_vs:
            with patch("tools.get_tools.get_mcp_tools", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_web_search_tool", new_callable=AsyncMock
                ):
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ):
                        mock_get_vs.return_value = mock_vs_tool

                        result = await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=False,
                            user_vs_files=False,
                            user_vs_memories=True,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        mock_get_vs.assert_called_once_with(
                            session=mock_session,
                            user_id=user_id,
                            scope="user_vs_memories",
                            pr=mock_pr,
                        )
                        assert result == [mock_vs_tool]

    async def test_preserves_tool_order(self):
        """Test that tools are returned in a consistent order."""
        user_id = uuid.uuid4()
        mcp_ids = [uuid.uuid4()]
        mock_session = AsyncMock()
        mock_pr = MagicMock()

        mock_mcp_tool = MagicMock(name="mcp_tool")
        mock_web_tool = MagicMock(name="web_tool")
        mock_vs_files_tool = MagicMock(name="vs_files_tool")
        mock_vs_memories_tool = MagicMock(name="vs_memories_tool")
        mock_skill_tool = MagicMock(name="skill_tool")

        with patch(
            "tools.get_tools.get_mcp_tools", new_callable=AsyncMock
        ) as mock_get_mcp:
            with patch(
                "tools.get_tools.get_web_search_tool", new_callable=AsyncMock
            ) as mock_get_web:
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ) as mock_get_vs:
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ) as mock_get_skill:
                        mock_get_mcp.return_value = [mock_mcp_tool]
                        mock_get_web.return_value = [mock_web_tool]
                        mock_get_vs.side_effect = [
                            mock_vs_files_tool,
                            mock_vs_memories_tool,
                        ]
                        mock_get_skill.return_value = mock_skill_tool

                        result = await get_agent_tools(
                            mcp_ids=mcp_ids,
                            session=mock_session,
                            web_search=True,
                            user_vs_files=True,
                            user_vs_memories=True,
                            user_vs_skills=True,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        assert result[0] == mock_mcp_tool
                        assert result[1] == mock_web_tool
                        assert result[2] == mock_vs_files_tool
                        assert result[3] == mock_vs_memories_tool
                        assert result[4] == mock_skill_tool

    async def test_handles_multiple_mcp_tools(self):
        """Test handling multiple MCP tools returned from get_mcp_tools."""
        user_id = uuid.uuid4()
        mcp_ids = [uuid.uuid4(), uuid.uuid4()]
        mock_session = AsyncMock()
        mock_pr = MagicMock()

        mock_mcp_tool1 = MagicMock()
        mock_mcp_tool2 = MagicMock()

        with patch(
            "tools.get_tools.get_mcp_tools", new_callable=AsyncMock
        ) as mock_get_mcp:
            with patch("tools.get_tools.get_web_search_tool", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ):
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ):
                        mock_get_mcp.return_value = [mock_mcp_tool1, mock_mcp_tool2]

                        result = await get_agent_tools(
                            mcp_ids=mcp_ids,
                            session=mock_session,
                            web_search=False,
                            user_vs_files=False,
                            user_vs_memories=False,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        assert len(result) == 2
                        assert mock_mcp_tool1 in result
                        assert mock_mcp_tool2 in result

    async def test_includes_skill_discovery_tool_when_enabled(self):
        """Test that skill discovery tool is included when user_vs_skills is True."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()
        mock_skill_tool = MagicMock()

        with patch("tools.get_tools.get_mcp_tools", new_callable=AsyncMock):
            with patch("tools.get_tools.get_web_search_tool", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ):
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ) as mock_get_skill:
                        mock_get_skill.return_value = mock_skill_tool

                        result = await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=False,
                            user_vs_files=False,
                            user_vs_memories=False,
                            user_vs_skills=True,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        assert mock_skill_tool in result
                        assert len(result) == 1

    async def test_skips_skill_discovery_tool_when_disabled(self):
        """Test that skill discovery tool is not included when user_vs_skills is False."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()

        with patch("tools.get_tools.get_mcp_tools", new_callable=AsyncMock):
            with patch("tools.get_tools.get_web_search_tool", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ):
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ) as mock_get_skill:
                        result = await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=False,
                            user_vs_files=False,
                            user_vs_memories=False,
                            user_vs_skills=False,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        mock_get_skill.assert_not_called()
                        assert len(result) == 0

    async def test_passes_correct_parameters_to_skill_discovery_tool(self):
        """Test that correct parameters are passed to get_user_skill_discovery_tool."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()
        mock_skill_tool = MagicMock()

        with patch("tools.get_tools.get_mcp_tools", new_callable=AsyncMock):
            with patch("tools.get_tools.get_web_search_tool", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ):
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ) as mock_get_skill:
                        mock_get_skill.return_value = mock_skill_tool

                        await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=False,
                            user_vs_files=False,
                            user_vs_memories=False,
                            user_vs_skills=True,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        mock_get_skill.assert_called_once_with(
                            session=mock_session,
                            user_id=user_id,
                            scope="user_vs_skills",
                            pr=mock_pr,
                        )

    async def test_only_user_vs_skills_enabled(self):
        """Test when only user_vs_skills is enabled."""
        user_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_pr = MagicMock()
        mock_skill_tool = MagicMock()

        with patch("tools.get_tools.get_mcp_tools", new_callable=AsyncMock):
            with patch("tools.get_tools.get_web_search_tool", new_callable=AsyncMock):
                with patch(
                    "tools.get_tools.get_vs_search_tool", new_callable=AsyncMock
                ):
                    with patch(
                        "tools.get_tools.get_user_skill_discovery_tool",
                        new_callable=AsyncMock,
                    ) as mock_get_skill:
                        mock_get_skill.return_value = mock_skill_tool

                        result = await get_agent_tools(
                            mcp_ids=[],
                            session=mock_session,
                            web_search=False,
                            user_vs_files=False,
                            user_vs_memories=False,
                            user_vs_skills=True,
                            user_id=user_id,
                            pr=mock_pr,
                        )

                        assert result == [mock_skill_tool]
                        mock_get_skill.assert_called_once()
