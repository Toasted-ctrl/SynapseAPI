import uuid

import pytest
from pydantic import ValidationError

from api.v1.inference.models import ClientToolCreate, ConversationCreate

MODEL = {"provider": "openai", "name": "gpt"}
SCHEMA = {"type": "object", "properties": {"command": {"type": "string"}}}
BASH = {"name": "bash", "description": "Runs a command.", "input_schema": SCHEMA}
RESULT = {"tool_call_id": "c1", "content": "out"}


class TestConversationCreate:
    def test_prompt_with_client_tools(self):
        payload = ConversationCreate(prompt="hi", model=MODEL, client_tools=[BASH])

        assert payload.client_tools[0].name == "bash"
        assert payload.tool_results is None

    def test_client_tools_allowed_with_agent_id(self):
        payload = ConversationCreate(
            prompt="hi", agent_id=uuid.uuid4(), client_tools=[BASH]
        )

        assert payload.client_tools is not None

    def test_tool_results_with_thread_id(self):
        payload = ConversationCreate(
            thread_id=uuid.uuid4(), model=MODEL, tool_results=[RESULT]
        )

        assert payload.prompt is None
        assert payload.tool_results[0].is_error is False

    @pytest.mark.parametrize(
        "fields",
        [
            {},
            {"prompt": "hi", "tool_results": [RESULT], "thread_id": uuid.uuid4()},
        ],
    )
    def test_requires_exactly_one_of_prompt_and_tool_results(self, fields):
        with pytest.raises(ValidationError, match="exactly one of prompt"):
            ConversationCreate(model=MODEL, **fields)

    def test_tool_results_require_thread_id(self):
        with pytest.raises(ValidationError, match="thread_id is required"):
            ConversationCreate(model=MODEL, tool_results=[RESULT])

    def test_tool_results_must_not_be_empty(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            ConversationCreate(model=MODEL, thread_id=uuid.uuid4(), tool_results=[])

    def test_client_tool_names_must_be_unique(self):
        with pytest.raises(ValidationError, match="unique"):
            ConversationCreate(prompt="hi", model=MODEL, client_tools=[BASH, BASH])


class TestClientToolCreate:
    @pytest.mark.parametrize("name", ["", "has space", "a" * 65, "semi;colon"])
    def test_rejects_invalid_names(self, name):
        with pytest.raises(ValidationError):
            ClientToolCreate(name=name, description="d", input_schema=SCHEMA)

    def test_rejects_non_object_schema(self):
        with pytest.raises(ValidationError, match="type 'object'"):
            ClientToolCreate(
                name="bash", description="d", input_schema={"type": "string"}
            )
