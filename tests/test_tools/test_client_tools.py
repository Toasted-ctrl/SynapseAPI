from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from api.v1.inference.models import ClientToolCreate
from tools.client_tools import (
    ClientToolCalledError,
    ClientToolsMiddleware,
    build_client_tool,
)

BASH = ClientToolCreate(
    name="bash",
    description="Runs a shell command on the user's machine.",
    input_schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)


def _request(name: str, args: dict | None = None, call_id: str = "call_1"):
    request = MagicMock()
    request.tool_call = {"name": name, "args": args or {}, "id": call_id}
    return request


class TestBuildClientTool:
    def test_uses_client_schema(self):
        tool = build_client_tool(BASH)

        assert tool.name == "bash"
        assert tool.description == BASH.description
        assert tool.args_schema == BASH.input_schema

    @pytest.mark.asyncio
    async def test_raises_if_run_on_server(self):
        tool = build_client_tool(BASH)

        with pytest.raises(ClientToolCalledError):
            await tool.coroutine(command="ls")


class TestClientToolsMiddleware:
    def test_registers_tools(self):
        middleware = ClientToolsMiddleware([BASH])

        assert [t.name for t in middleware.tools] == ["bash"]

    @pytest.mark.asyncio
    async def test_passes_server_tools_to_handler(self):
        middleware = ClientToolsMiddleware([BASH])
        handler = AsyncMock(return_value="server result")

        with patch("tools.client_tools.interrupt") as interrupt:
            result = await middleware.awrap_tool_call(_request("web_search"), handler)

        assert result == "server result"
        interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_interrupts_on_client_tool_and_returns_result(self):
        middleware = ClientToolsMiddleware([BASH])
        handler = AsyncMock()

        with patch(
            "tools.client_tools.interrupt",
            return_value={"content": "file.txt", "is_error": False},
        ) as interrupt:
            result = await middleware.awrap_tool_call(
                _request("bash", {"command": "ls"}), handler
            )

        interrupt.assert_called_once_with(
            {"tool_call_id": "call_1", "name": "bash", "args": {"command": "ls"}}
        )
        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.content == "file.txt"
        assert result.tool_call_id == "call_1"
        assert result.name == "bash"
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_error_result_has_error_status(self):
        middleware = ClientToolsMiddleware([BASH])

        with patch(
            "tools.client_tools.interrupt",
            return_value={"content": "declined", "is_error": True},
        ):
            result = await middleware.awrap_tool_call(_request("bash"), AsyncMock())

        assert result.status == "error"
