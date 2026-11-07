import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from api.v1.inference.models import ClientToolCreate, ToolResultCreate
from api.v1.inference.v1_conversation import _serialize_event
from providers.agent import (
    InvalidToolResultsError,
    NoPendingToolCallsError,
    PendingToolCallsError,
    _drop_empty_text_blocks,
    _without_empty_text,
    build_agent_input,
    build_agent_model,
    run_agent,
    stream_agent,
)

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


# Client tool calls, run through a real agent graph with an in-memory checkpointer


class _FakeToolModel(GenericFakeChatModel):
    """A fake chat model that accepts tools, replying with the given messages."""

    # GenericFakeChatModel can't stream a message without text content
    disable_streaming: bool = True

    def bind_tools(self, tools, **kwargs):
        return self


BASH = ClientToolCreate(
    name="bash",
    description="Runs a shell command on the user's machine.",
    input_schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)


def _build_agent(replies: list[AIMessage], server_calls: list[str]):
    @tool
    def server_tool(query: str) -> str:
        """Looks something up on the server."""
        server_calls.append(query)
        return f"server says {query}"

    model = _FakeToolModel(messages=iter(replies))
    with patch("providers.agent.build_llm", return_value=model):
        return build_agent_model(
            langchain_con=MagicMock(),
            model="fake",
            base_url="",
            temperature=None,
            top_k=None,
            top_p=None,
            encrypted_api_key=None,
            tools=[server_tool],
            checkpointer=InMemorySaver(),
            client_tools=[BASH],
        )


def _mixed_tool_call_replies() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "server_tool", "args": {"query": "q"}, "id": "s1"},
                {"name": "bash", "args": {"command": "ls"}, "id": "c1"},
            ],
        ),
        AIMessage(content="All done."),
    ]


@pytest.fixture
def no_langfuse():
    with patch("providers.agent.langfuse_handler", BaseCallbackHandler()):
        yield


@pytest.mark.usefixtures("no_langfuse")
class TestClientToolRun:
    @pytest.mark.asyncio
    async def test_pauses_on_client_tool_and_resumes_with_result(self):
        server_calls: list[str] = []
        agent = _build_agent(_mixed_tool_call_replies(), server_calls)
        thread_id, user_id = uuid.uuid4(), uuid.uuid4()

        agent_input = await build_agent_input(agent, thread_id, "hi", None)
        result = await run_agent(agent, agent_input, thread_id, user_id)

        assert result["pending_tool_calls"] == [
            {"tool_call_id": "c1", "name": "bash", "args": {"command": "ls"}}
        ]
        assert server_calls == ["q"]

        agent_input = await build_agent_input(
            agent,
            thread_id,
            None,
            [ToolResultCreate(tool_call_id="c1", content="file.txt")],
        )
        assert isinstance(agent_input, Command)
        result = await run_agent(agent, agent_input, thread_id, user_id)

        assert result["pending_tool_calls"] == []
        # The server tool ran before the pause and isn't run again
        assert server_calls == ["q"]
        tool_messages = {
            m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)
        }
        assert tool_messages["s1"].content == "server says q"
        assert tool_messages["c1"].content == "file.txt"
        assert tool_messages["c1"].status == "success"
        assert result["messages"][-1].content == "All done."

    @pytest.mark.asyncio
    async def test_stream_emits_tool_request(self):
        agent = _build_agent(_mixed_tool_call_replies(), [])
        thread_id, user_id = uuid.uuid4(), uuid.uuid4()

        agent_input = await build_agent_input(agent, thread_id, "hi", None)
        events = [e async for e in stream_agent(agent, agent_input, thread_id, user_id)]

        requests = [e for e in events if e["type"] == "tool_request"]
        assert requests == [
            {
                "type": "tool_request",
                "data": {
                    "tool_calls": [
                        {
                            "tool_call_id": "c1",
                            "name": "bash",
                            "args": {"command": "ls"},
                        }
                    ]
                },
            }
        ]
        # Every event can be sent to the client as JSON
        for event in events:
            json.dumps(_serialize_event(event))

    @pytest.mark.asyncio
    async def test_prompt_while_paused_raises(self):
        agent = _build_agent(_mixed_tool_call_replies(), [])
        thread_id, user_id = uuid.uuid4(), uuid.uuid4()
        await run_agent(
            agent,
            await build_agent_input(agent, thread_id, "hi", None),
            thread_id,
            user_id,
        )

        with pytest.raises(PendingToolCallsError) as e:
            await build_agent_input(agent, thread_id, "hello again", None)

        assert [c["tool_call_id"] for c in e.value.pending] == ["c1"]

    @pytest.mark.asyncio
    async def test_results_without_pause_raise(self):
        agent = _build_agent([AIMessage(content="Hi.")], [])

        with pytest.raises(NoPendingToolCallsError):
            await build_agent_input(
                agent,
                uuid.uuid4(),
                None,
                [ToolResultCreate(tool_call_id="c1", content="x")],
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "results, message",
        [
            ([], "Missing results"),
            (["c1", "other"], "Unknown tool_call_id"),
            (["c1", "c1"], "twice"),
        ],
    )
    async def test_mismatched_results_raise(self, results, message):
        agent = _build_agent(_mixed_tool_call_replies(), [])
        thread_id, user_id = uuid.uuid4(), uuid.uuid4()
        await run_agent(
            agent,
            await build_agent_input(agent, thread_id, "hi", None),
            thread_id,
            user_id,
        )
        tool_results = [ToolResultCreate(tool_call_id=r, content="x") for r in results]

        with pytest.raises(InvalidToolResultsError, match=message):
            await build_agent_input(agent, thread_id, None, tool_results)
