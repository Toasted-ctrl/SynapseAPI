from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from api.v1.playground.models import AgentModel, AgentTools


# Conversation models
class ClientToolCreate(BaseModel):
    """A tool that the client runs itself, such as a shell command on the user's machine.

    The model can call it like any other tool. Instead of running it, the API pauses
    the run and returns the call to the client, which sends the result back with
    tool_results to continue.
    """

    name: Annotated[
        str,
        Field(
            pattern=r"^[a-zA-Z0-9_-]{1,64}$",
            description="Name of the tool, unique among all tools of the request.",
        ),
    ]
    description: Annotated[
        str, Field(min_length=1, description="Tells the model what the tool does.")
    ]
    input_schema: Annotated[
        dict[str, Any],
        Field(description="JSON Schema of the tool's arguments. Must be an object."),
    ]

    @model_validator(mode="after")
    def validate_object_schema(self):
        if self.input_schema.get("type") != "object":
            raise ValueError("input_schema must be a JSON Schema with type 'object'.")
        return self


class ToolResultCreate(BaseModel):
    """The result of a client tool call, sent to continue a paused run."""

    tool_call_id: Annotated[
        str, Field(min_length=1, description="ID of the tool call this result is for.")
    ]
    content: Annotated[
        str, Field(description="Output of the tool, given to the model.")
    ]
    is_error: Annotated[
        bool,
        Field(description="True if the tool failed or the user declined to run it."),
    ] = False


class ToolCallRead(BaseModel):
    """A client tool call the model made, waiting for its result."""

    tool_call_id: Annotated[str, Field(description="ID to send the result back with.")]
    name: Annotated[str, Field(description="Name of the client tool.")]
    args: Annotated[dict[str, Any], Field(description="Arguments for the tool.")]


class ConversationCreate(BaseModel):
    """Request body for a conversation turn.

    Either references a stored Agent by agent_id, or configures the model and tools
    inline. The inline fields reuse the Agent configuration models, so both paths
    behave the same. When agent_id is given, the stored Agent's system prompt is used.

    A turn either starts with a prompt, or continues a run that is paused on client
    tool calls by sending their tool_results.
    """

    agent_id: Annotated[
        UUID | None,
        Field(
            description="ID of a stored Agent to converse with. Excludes model and tools."
        ),
    ] = None
    thread_id: Annotated[
        UUID | None,
        Field(
            description="ID of an existing thread to continue. A new thread is started if omitted."
        ),
    ] = None
    prompt: Annotated[
        str | None,
        Field(description="The user's message. Excludes tool_results."),
    ] = None
    tool_results: Annotated[
        list[ToolResultCreate] | None,
        Field(
            description=(
                "Results of the pending client tool calls of thread_id, to continue "
                "the paused run. Excludes prompt."
            )
        ),
    ] = None
    client_tools: Annotated[
        list[ClientToolCreate] | None,
        Field(
            description=(
                "Tools the client runs itself. Send the same tools again when "
                "sending tool_results. Allowed together with agent_id."
            )
        ),
    ] = None

    model: Annotated[
        AgentModel | None,
        Field(
            description="The model to converse with. Required when agent_id is omitted."
        ),
    ] = None
    tools: Annotated[
        AgentTools | None, Field(description="Tools the model is allowed to use.")
    ] = None
    system_prompt: Annotated[
        str | None,
        Field(description="System prompt given to the model. Excluded with agent_id."),
    ] = None
    stop_sequence: Annotated[
        list[str] | None,
        Field(
            description="Sequences that stop generation when produced. Excluded with agent_id."
        ),
    ] = None

    @model_validator(mode="after")
    def validate_agent_or_model(self):
        if self.agent_id is not None:
            inline = [
                f
                for f in ("model", "tools", "system_prompt", "stop_sequence")
                if getattr(self, f) is not None
            ]
            if inline:
                raise ValueError(
                    f"When agent_id is provided, the following fields must be omitted: {', '.join(inline)}."
                )
        elif self.model is None:
            raise ValueError("When agent_id is not provided, model is required.")
        return self

    @model_validator(mode="after")
    def validate_prompt_or_tool_results(self):
        if (self.prompt is None) == (self.tool_results is None):
            raise ValueError("Provide exactly one of prompt or tool_results.")
        if self.tool_results is not None:
            if self.thread_id is None:
                raise ValueError("thread_id is required when sending tool_results.")
            if not self.tool_results:
                raise ValueError("tool_results must not be empty.")
        return self

    @model_validator(mode="after")
    def validate_unique_client_tools(self):
        names = [t.name for t in self.client_tools or []]
        if len(names) != len(set(names)):
            raise ValueError("client_tools names must be unique.")
        return self


class ConversationRead(BaseModel):
    """A conversation turn as returned by the API."""

    thread_id: Annotated[
        UUID, Field(description="ID of the thread the turn belongs to.")
    ]
    messages: Annotated[list, Field(description="Messages in the thread.")]
    pending_tool_calls: Annotated[
        list[ToolCallRead],
        Field(
            description=(
                "Client tool calls the run is paused on. Run them and send their "
                "results with tool_results to continue."
            )
        ),
    ] = Field(default_factory=list)


# Decision models
class DecisionCreate(BaseModel):
    """Request body for a decision by a stored Decision Agent."""

    agent_id: Annotated[UUID, Field(description="ID of the stored Decision Agent.")]
    state: Annotated[
        str | dict[str, str],
        Field(
            description=(
                "The situation to decide on. A string if the Agent's state type is "
                "'text', or an object of field name to value if it is 'fields'."
            )
        ),
    ]


class DecisionRead(BaseModel):
    """A decision as returned by the API."""

    agent_id: Annotated[UUID, Field(description="ID of the Decision Agent.")]
    answers: Annotated[
        dict[str, dict[str, Any]],
        Field(
            description="Answers keyed by question name, as returned by the provider."
        ),
    ]


# Translation models
class TranslationModel(BaseModel):
    """The model used to translate the prompt."""

    provider: Annotated[
        str, Field(min_length=1, description="Model provider, e.g. 'Ollama-1'.")
    ]
    name: Annotated[
        str,
        Field(
            min_length=1,
            description="Provider-specific translation model name, e.g. 'translategemma:latest'.",
        ),
    ]


class TranslationParameters(BaseModel):
    temperature: Annotated[
        float, Field(ge=0.0, le=2.0, description="Controls randomness of the output.")
    ] = 0.1


class TranslationCreate(BaseModel):
    model: Annotated[
        TranslationModel, Field(description="The translation model to use.")
    ]
    from_lang_code: Annotated[str, Field(description="Language code of the prompt.")]
    to_lang_code: Annotated[str, Field(description="Language code to translate to.")]
    parameters: Annotated[
        TranslationParameters, Field(description="Sampling parameters for the model.")
    ] = Field(default_factory=TranslationParameters)
    prompt: Annotated[str, Field(min_length=1, description="The text to translate.")]


class TranslationRead(BaseModel):
    model: TranslationModel
    prompt: str
    translation: str
    from_lang_code: str
    to_lang_code: str


class TranslationLanguagesRead(BaseModel):
    languages: list[str]


# Vector embedding models
class VectorEmbeddingModel(BaseModel):
    """The embedding model used to generate the vector embedding."""

    provider: Annotated[
        str, Field(min_length=1, description="Model provider, e.g. 'openai'.")
    ]
    name: Annotated[
        str, Field(min_length=1, description="Provider-specific embedding model name.")
    ]


class VectorEmbeddingCreate(BaseModel):
    model: Annotated[
        VectorEmbeddingModel, Field(description="The embedding model to use.")
    ]
    prompt: Annotated[str, Field(min_length=1, description="The text to embed.")]
    dimensions: Annotated[
        int | None,
        Field(
            gt=0,
            description="Requested embedding size. Uses the model's default if omitted; not all providers support it.",
        ),
    ] = None


class VectorEmbeddingRead(BaseModel):
    model: VectorEmbeddingModel
    prompt: str
    dimensions: int
    embedding: list[float]
