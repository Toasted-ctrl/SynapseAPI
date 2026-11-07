"""Tools that the client runs itself, such as shell commands on the user's machine.

The tools are bound to the model like any other tool. When the model calls one,
ClientToolsMiddleware pauses the run with a LangGraph interrupt instead of running
it. The run is saved by the checkpointer, the call is returned to the client, and
the client continues the run by resuming with the tool's result.

Each tool call runs as its own task, so server tools called in the same model
response still run, and are not run again when the run continues.
"""

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command, interrupt

from api.v1.inference.models import ClientToolCreate


class ClientToolCall(TypedDict):
    """A client tool call, as stored in the interrupt."""

    tool_call_id: str
    name: str
    args: dict[str, Any]


class ClientToolResult(TypedDict):
    """The value a paused client tool call is resumed with."""

    content: str
    is_error: bool


class ClientToolCalledError(RuntimeError):
    """Raised if a client tool is run on the server, which must never happen."""


def build_client_tool(spec: ClientToolCreate) -> StructuredTool:
    """Returns a tool with the client's schema, so it can be bound to the model.

    The tool is never run: ClientToolsMiddleware intercepts its calls.
    """

    async def _never_run(**_: Any) -> str:
        raise ClientToolCalledError(f"Client tool '{spec.name}' was run on the server.")

    return StructuredTool(
        name=spec.name,
        description=spec.description,
        args_schema=spec.input_schema,
        coroutine=_never_run,
    )


class ClientToolsMiddleware(AgentMiddleware):
    """Registers the client's tools and pauses the run when the model calls one."""

    def __init__(self, client_tools: list[ClientToolCreate]):
        super().__init__()
        self.tools: list[BaseTool] = [build_client_tool(s) for s in client_tools]
        self._names = {t.name for t in self.tools}

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        if tool_call["name"] not in self._names:
            return await handler(request)

        tool_call_id = tool_call["id"]
        if not tool_call_id:
            # The client sends the result back by this ID, so it can't be missing
            raise ValueError(f"Call of client tool '{tool_call['name']}' has no ID.")

        call: ClientToolCall = {
            "tool_call_id": tool_call_id,
            "name": tool_call["name"],
            "args": tool_call.get("args") or {},
        }
        result: ClientToolResult = interrupt(call)
        return ToolMessage(
            content=result["content"],
            name=tool_call["name"],
            tool_call_id=tool_call_id,
            status="error" if result["is_error"] else "success",
        )
