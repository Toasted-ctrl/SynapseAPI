from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from providers.agent import _drop_empty_text_blocks, _without_empty_text

THINKING = {"type": "thinking", "thinking": "", "signature": "sig", "index": 0}


class TestWithoutEmptyText:
    """Tests for the _without_empty_text function."""

    def test_removes_blank_str_items_from_ai_list_content(self):
        message = AIMessage(content=["", THINKING, "  \n", "Hello"])

        result = _without_empty_text(message)

        assert result.content == [THINKING, "Hello"]
        # The original (checkpointed) message is left unchanged
        assert message.content == ["", THINKING, "  \n", "Hello"]

    def test_returns_same_message_when_nothing_to_remove(self):
        message = AIMessage(content=[THINKING, "Hello"])

        assert _without_empty_text(message) is message

    def test_ignores_str_content(self):
        message = AIMessage(content="")

        assert _without_empty_text(message) is message

    def test_ignores_non_ai_messages(self):
        message = HumanMessage(content=["", "Hello"])

        assert _without_empty_text(message) is message


class TestDropEmptyTextBlocks:
    """Tests for the _drop_empty_text_blocks middleware."""

    @pytest.mark.asyncio
    async def test_passes_cleaned_messages_to_handler(self):
        messages = [
            HumanMessage(content="list files"),
            AIMessage(content=["", THINKING, "<bash>\nls\n"]),
            HumanMessage(content="<bash_output></bash_output>"),
        ]
        request = MagicMock()
        request.messages = messages
        handler = AsyncMock(return_value="response")

        result = await _drop_empty_text_blocks.awrap_model_call(request, handler)

        assert result == "response"
        request.override.assert_called_once()
        cleaned = request.override.call_args.kwargs["messages"]
        assert cleaned[0] is messages[0]
        assert cleaned[1].content == [THINKING, "<bash>\nls\n"]
        assert cleaned[2] is messages[2]
        handler.assert_awaited_once_with(request.override.return_value)
