from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from api.v1.playground.models import AgentModel, AgentTools


# Conversation models
class ConversationCreate(BaseModel):
    """Request body for a conversation turn.

    Either references a stored Agent by agent_id, or configures the model and tools
    inline. The inline fields reuse the Agent configuration models, so both paths
    behave the same. A system prompt is only used when it comes from a stored Agent.
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
    prompt: Annotated[str, Field(description="The user's message.")]

    model: Annotated[
        AgentModel | None,
        Field(
            description="The model to converse with. Required when agent_id is omitted."
        ),
    ] = None
    tools: Annotated[
        AgentTools | None, Field(description="Tools the model is allowed to use.")
    ] = None

    @model_validator(mode="after")
    def validate_agent_or_model(self):
        if self.agent_id is not None:
            inline = [f for f in ("model", "tools") if getattr(self, f) is not None]
            if inline:
                raise ValueError(
                    f"When agent_id is provided, the following fields must be omitted: {', '.join(inline)}."
                )
        elif self.model is None:
            raise ValueError("When agent_id is not provided, model is required.")
        return self


class ConversationRead(BaseModel):
    """A conversation turn as returned by the API."""

    thread_id: Annotated[
        UUID, Field(description="ID of the thread the turn belongs to.")
    ]
    messages: Annotated[list, Field(description="Messages in the thread.")]


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
