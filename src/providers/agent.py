import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph.state import CompiledStateGraph

from core.langfuse import langfuse_handler
from core.logging import get_logger
from providers.chat_completion import _build_llm
from providers.dataclasses import LangChainCon

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
    checkpointer: AsyncPostgresSaver | None = None,
) -> CompiledStateGraph:

    log.debug("Creating agent ...")

    model = _build_llm(
        langchain_con=langchain_con,
        model=model,
        base_url=base_url,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        encrypted_api_key=encrypted_api_key,
    )

    log.debug("Built Model to add to agent ...")

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=[_tool_errors_to_model],
    )

    log.debug("Agent created, returning ...")

    return agent


class AgentEvent(TypedDict):
    type: Literal["thread", "message", "tool_call", "tool_result", "update", "error"]
    data: dict[str, Any]


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
    agent: CompiledStateGraph, prompt: str, thread_id: uuid.UUID, user_id: uuid.UUID
) -> AsyncGenerator[AgentEvent]:

    config = {
        "configurable": {"thread_id": str(thread_id)},
        "callbacks": [langfuse_handler],
        "metadata": {"thread_id": str(thread_id), "user_id": str(user_id)},
    }

    log.debug(f"Stream agent config created for User '{user_id}'.")

    yield {"type": "thread", "data": {"thread_id": str(thread_id)}}
    async for event in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
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
            yield {"type": "update", "data": event["data"]}


async def run_agent(
    agent: CompiledStateGraph, prompt: str, thread_id: uuid.UUID, user_id: uuid.UUID
) -> dict:
    config = {
        "configurable": {"thread_id": str(thread_id)},
        "callbacks": [langfuse_handler],
        "metadata": {"thread_id": str(thread_id), "user_id": str(user_id)},
    }

    log.debug(f"Run agent config created for User '{user_id}'.")

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]}, config=config
    )

    return result
