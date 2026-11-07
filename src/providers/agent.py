import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from api.v1.inference.models import ClientToolCreate, ToolResultCreate
from core.langfuse import langfuse_handler
from core.logging import get_logger
from providers.dataclasses import LangChainCon
from providers.llm import build_llm
from tools.client_tools import ClientToolCall, ClientToolResult, ClientToolsMiddleware

log = get_logger()


# TODO: Create tests for all functions here.


@wrap_tool_call
async def _tool_errors_to_model(request, handler):
    """Returns any tool exception (e.g. MCP protocol/transport errors) to the model
    as an error ToolMessage, instead of letting it abort the whole agent run."""
    try:
        return await handler(request)
    except GraphBubbleUp:
        # LangGraph control flow (interrupts etc.), must propagate.
        raise
    except Exception as e:
        tool_name = request.tool_call["name"]
        log.warning(f"Tool '{tool_name}' raised an error: {e!r}")
        return ToolMessage(
            content=f"Tool '{tool_name}' failed with error: {e}",
            name=tool_name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )


def _without_empty_text(message: BaseMessage) -> BaseMessage:
    """Returns the message without blank str items in its list content.

    When a model returns thinking that wasn't requested, langchain_anthropic streams
    the reply as str chunks mixed with a thinking block, which merge into content
    like ["", {"type": "thinking", ...}, "text"]. Replaying that "" sends an empty
    text block, which Anthropic rejects with a 400.
    """
    if not isinstance(message, AIMessage) or not isinstance(message.content, list):
        return message
    content = [
        block
        for block in message.content
        if not (isinstance(block, str) and not block.strip())
    ]
    if len(content) == len(message.content):
        return message
    return message.model_copy(update={"content": content})


@wrap_model_call
async def _drop_empty_text_blocks(request, handler):
    """Removes blank text from stored AI messages before every model call.

    Only the request is changed, not the checkpoint, so threads that already
    contain such messages work again.
    """
    messages = [_without_empty_text(m) for m in request.messages]
    return await handler(request.override(messages=messages))


def build_agent_model(
    langchain_con: LangChainCon,
    model: str,
    base_url: str,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    encrypted_api_key: str | None,
    tools: list,
    system_prompt: str | None = None,
    stop_sequence: list[str] | None = None,
    checkpointer: AsyncPostgresSaver | None = None,
    client_tools: list[ClientToolCreate] | None = None,
) -> CompiledStateGraph:

    log.debug("Creating agent ...")

    llm = build_llm(
        langchain_con=langchain_con,
        model=model,
        base_url=base_url,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        encrypted_api_key=encrypted_api_key,
        stop_sequence=stop_sequence,
    )

    log.debug("Built Model to add to agent ...")

    # First in the list is the outermost layer, so _tool_errors_to_model sees (and
    # passes on) the interrupts of client tool calls.
    middleware = [_drop_empty_text_blocks, _tool_errors_to_model]
    if client_tools:
        middleware.append(ClientToolsMiddleware(client_tools))

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=middleware,
    )

    log.debug("Agent created, returning ...")

    return agent


def _run_config(thread_id: uuid.UUID, user_id: uuid.UUID) -> RunnableConfig:
    return {
        "configurable": {"thread_id": str(thread_id)},
        "callbacks": [langfuse_handler],
        "metadata": {"thread_id": str(thread_id), "user_id": str(user_id)},
    }


class AgentEvent(TypedDict):
    type: Literal[
        "thread",
        "message",
        "tool_call",
        "tool_result",
        "tool_request",
        "update",
        "error",
    ]
    data: dict[str, Any]


AgentInput = dict[str, Any] | Command

# Key of the interrupts in an "updates" stream event and in an ainvoke result
INTERRUPT_KEY = "__interrupt__"


class PendingToolCallsError(Exception):
    """The run is paused on client tool calls, so it can't take a new prompt."""

    def __init__(self, pending: list[ClientToolCall]):
        super().__init__("The thread is waiting for the results of client tool calls.")
        self.pending = pending


class NoPendingToolCallsError(Exception):
    """Tool results were sent, but the run isn't paused on any client tool calls."""

    def __init__(self):
        super().__init__("The thread has no pending client tool calls.")


class InvalidToolResultsError(Exception):
    """The tool results don't match the pending client tool calls."""


def _client_tool_calls(interrupts: tuple[Interrupt, ...]) -> list[ClientToolCall]:
    return [i.value for i in interrupts]


async def build_agent_input(
    agent: CompiledStateGraph,
    thread_id: uuid.UUID,
    prompt: str | None,
    tool_results: list[ToolResultCreate] | None,
) -> AgentInput:
    """Returns the input that starts a turn with the prompt, or continues the run
    paused on client tool calls with their results.

    Raises PendingToolCallsError for a prompt while the run is paused,
    NoPendingToolCallsError for results while it isn't, and InvalidToolResultsError
    if the results don't match the pending calls exactly.
    """
    state = await agent.aget_state({"configurable": {"thread_id": str(thread_id)}})
    interrupts: tuple[Interrupt, ...] = state.interrupts

    if tool_results is None:
        if interrupts:
            raise PendingToolCallsError(_client_tool_calls(interrupts))
        return {"messages": [{"role": "user", "content": prompt}]}

    if not interrupts:
        raise NoPendingToolCallsError()

    results = {r.tool_call_id: r for r in tool_results}
    if len(results) != len(tool_results):
        raise InvalidToolResultsError("tool_results contains a tool_call_id twice.")
    pending = {i.value["tool_call_id"]: i for i in interrupts}
    if unknown := results.keys() - pending.keys():
        raise InvalidToolResultsError(
            f"Unknown tool_call_id(s): {', '.join(sorted(unknown))}."
        )
    if missing := pending.keys() - results.keys():
        raise InvalidToolResultsError(
            f"Missing results for tool_call_id(s): {', '.join(sorted(missing))}."
        )

    resume: dict[str, ClientToolResult] = {
        pending[call_id].id: {"content": r.content, "is_error": r.is_error}
        for call_id, r in results.items()
    }
    return Command(resume=resume)


def _extract_content(token: Any) -> str:
    if hasattr(token, "text") and token.text:
        return token.text
    content = getattr(token, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


async def stream_agent(
    agent: CompiledStateGraph,
    agent_input: AgentInput,
    thread_id: uuid.UUID,
    user_id: uuid.UUID,
) -> AsyncGenerator[AgentEvent]:

    config = _run_config(thread_id=thread_id, user_id=user_id)

    log.debug(f"Stream agent config created for User '{user_id}'.")

    yield {"type": "thread", "data": {"thread_id": str(thread_id)}}
    async for event in agent.astream(
        agent_input,
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if event["type"] == "messages":
            token, metadata = event["data"]

            if isinstance(token, AIMessageChunk):
                if token.tool_call_chunks:
                    for chunk in token.tool_call_chunks:
                        yield {
                            "type": "tool_call",
                            "data": {
                                "name": chunk.get("name"),
                                "args": chunk.get("args", ""),
                                "id": chunk.get("id"),
                                "index": chunk.get("index", 0),
                                "metadata": metadata,
                            },
                        }

                content = _extract_content(token)
                if content:
                    yield {
                        "type": "message",
                        "data": {"content": content, "metadata": metadata},
                    }

            elif isinstance(token, ToolMessage):
                yield {
                    "type": "tool_result",
                    "data": {
                        "content": token.content,
                        "name": token.name,
                        "tool_call_id": token.tool_call_id,
                        "metadata": metadata,
                    },
                }

        elif event["type"] == "updates":
            data = event["data"]
            # The run paused on client tool calls. The stream ends after this event,
            # and the client continues the run by sending the results.
            if isinstance(data, dict) and INTERRUPT_KEY in data:
                yield {
                    "type": "tool_request",
                    "data": {"tool_calls": _client_tool_calls(data[INTERRUPT_KEY])},
                }
            else:
                yield {"type": "update", "data": data}


class AgentRunResult(TypedDict):
    messages: list[BaseMessage]
    # Client tool calls the run is paused on; empty if the run finished
    pending_tool_calls: list[ClientToolCall]


async def run_agent(
    agent: CompiledStateGraph,
    agent_input: AgentInput,
    thread_id: uuid.UUID,
    user_id: uuid.UUID,
) -> AgentRunResult:
    config = _run_config(thread_id=thread_id, user_id=user_id)

    log.debug(f"Run agent config created for User '{user_id}'.")

    result = await agent.ainvoke(agent_input, config=config)

    return {
        "messages": result["messages"],
        "pending_tool_calls": _client_tool_calls(result.get(INTERRUPT_KEY, ())),
    }
