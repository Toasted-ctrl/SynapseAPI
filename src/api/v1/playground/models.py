from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class AgentType(StrEnum):
    """Kinds of Agent. Values must match the ck_agent_configs_agent_type constraint."""

    CONVERSATIONAL = "Conversational"
    DECISION = "Decision"


class AgentModelParameters(BaseModel):
    """Sampling parameters passed to the model provider."""

    temperature: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=2.0,
            description="Controls randomness. Lower values make output more focused and deterministic; higher values make it more varied.",
        ),
    ] = None
    top_p: Annotated[
        float | None,
        Field(
            gt=0.0,
            le=1.0,
            description="Nucleus sampling. The model samples only from the smallest set of tokens whose cumulative probability exceeds top_p. Lower values restrict output to more likely tokens.",
        ),
    ] = None
    top_k: Annotated[
        int | None,
        Field(
            ge=1,
            description="Restricts sampling to the k most likely tokens at each step. Lower values make output more predictable.",
        ),
    ] = None


class AgentModel(BaseModel):
    """The model the Agent runs on."""

    provider: Annotated[
        str, Field(min_length=1, description="Model provider, e.g. 'anthropic'.")
    ]
    name: Annotated[
        str, Field(min_length=1, description="Provider-specific model name.")
    ]
    parameters: Annotated[
        AgentModelParameters, Field(description="Sampling parameters for the model.")
    ] = Field(default_factory=AgentModelParameters)


class AgentTools(BaseModel):
    """Tools the Agent is allowed to use."""

    mcp_ids: Annotated[
        list[UUID],
        Field(description="List of MCP servers that Agent is allowed to use."),
    ] = Field(default_factory=list)
    web_search: Annotated[
        bool, Field(description="Allows the Agent to search the web.")
    ] = False
    user_vs_files: Annotated[
        bool, Field(description="Allows the Agent to search the user's files.")
    ] = False
    user_vs_memories: Annotated[
        bool, Field(description="Allows the Agent to search the user's memories.")
    ] = False
    user_vs_skills: Annotated[
        bool, Field(description="Allows the Agent to discover the user's skills.")
    ] = False


class AgentConfigurationBase(BaseModel):
    """Fields shared by all Agent configuration models.

    Everything except name and description is stored as JSON by database.agent, so
    changes here must follow the settings versioning rules described there.
    """

    name: Annotated[str, Field(min_length=1, description="Name of the Agent.")]
    description: Annotated[
        str | None, Field(description="Description of the Agent.")
    ] = None
    model: AgentModel
    system_prompt: Annotated[
        str | None, Field(description="System prompt given to the Agent.")
    ] = None
    tools: Annotated[
        AgentTools, Field(description="Tools the Agent is allowed to use.")
    ] = Field(default_factory=AgentTools)


class AgentConfigurationCreate(AgentConfigurationBase):
    """Request body for creating or updating an Agent. The server assigns the id."""


class AgentConfigurationRead(AgentConfigurationBase):
    """An Agent as returned by the API."""

    id: Annotated[UUID, Field(description="ID of the Agent.")]
    is_encrypted: Annotated[
        bool,
        Field(
            default=False,
            description="Whether this agent's configuration is stored encrypted.",
        ),
    ] = False


# Decision Agent models


class DecisionAgentModel(BaseModel):
    """The System One model the Decision Agent runs on."""

    provider: Annotated[
        str, Field(min_length=1, description="Model provider, e.g. 'anthropic'.")
    ]
    name: Annotated[
        str, Field(min_length=1, description="Provider-specific model name.")
    ]


class DecisionStateType(StrEnum):
    TEXT = "text"
    FIELDS = "fields"


class DecisionStateField(BaseModel):
    """A named value callers send as part of the state."""

    name: Annotated[str, Field(min_length=1, description="Name of the field.")]
    description: Annotated[
        str | None, Field(description="What the field contains.")
    ] = None


class DecisionState(BaseModel):
    """The situation callers send for the Decision Agent to decide on."""

    type: Annotated[
        DecisionStateType,
        Field(
            description="'text' if callers send state as a single string, 'fields' if they send named fields."
        ),
    ] = DecisionStateType.TEXT
    description: Annotated[
        str | None,
        Field(description="Describes the situation callers should send."),
    ] = None
    fields: Annotated[
        list[DecisionStateField],
        Field(description="Fields callers send. Required when type is 'fields'."),
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_fields(self) -> "DecisionState":
        if self.type == DecisionStateType.FIELDS and not self.fields:
            raise ValueError("fields are required when state type is 'fields'")
        if self.type == DecisionStateType.TEXT and self.fields:
            raise ValueError("fields are only allowed when state type is 'fields'")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("state field names must be unique")
        return self


class DecisionQuestionBase(BaseModel):
    name: Annotated[
        str,
        Field(
            min_length=1,
            description="Name of the question. Used as its key in the answers.",
        ),
    ]
    instructions: Annotated[
        str, Field(min_length=1, description="What the Agent should decide.")
    ]


class ChoiceQuestion(DecisionQuestionBase):
    """Pick one of a set of named options."""

    type: Literal["choice"] = "choice"
    options: Annotated[
        list[Annotated[str, Field(min_length=1)]],
        Field(min_length=2, description="Names of the options to pick from."),
    ]

    @model_validator(mode="after")
    def _check_options(self) -> "ChoiceQuestion":
        if len(self.options) != len(set(self.options)):
            raise ValueError("options must be unique")
        return self


class ScoreQuestion(DecisionQuestionBase):
    """Rate on an ordered scale. Each legend entry is a level, starting at 0."""

    type: Literal["score"] = "score"
    legend: Annotated[
        list[Annotated[str, Field(min_length=1)]],
        Field(
            min_length=2,
            description="Labels of the score levels, lowest first, e.g. ['routine', 'same-day', 'immediate'].",
        ),
    ]


class NoulQuestion(DecisionQuestionBase):
    """Answered with a single value between 0 and 1."""

    type: Literal["noul"] = "noul"


DecisionQuestion = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")
]


class DecisionAgentConfigurationBase(BaseModel):
    """Fields shared by all Decision Agent configuration models.

    Everything except name and description is stored as JSON by database.agent, so
    changes here must follow the settings versioning rules described there.
    """

    name: Annotated[str, Field(min_length=1, description="Name of the Agent.")]
    description: Annotated[
        str | None, Field(description="Description of the Agent.")
    ] = None
    model: DecisionAgentModel
    state: Annotated[DecisionState, Field(description="The state callers send.")] = (
        Field(default_factory=DecisionState)
    )
    questions: Annotated[
        list[DecisionQuestion],
        Field(min_length=1, description="Questions the Agent answers."),
    ]

    @model_validator(mode="after")
    def _check_question_names(self) -> "DecisionAgentConfigurationBase":
        names = [question.name for question in self.questions]
        if len(names) != len(set(names)):
            raise ValueError("question names must be unique")
        return self


class DecisionAgentConfigurationCreate(DecisionAgentConfigurationBase):
    """Request body for creating or updating a Decision Agent. The server assigns the id."""


class DecisionAgentConfigurationRead(DecisionAgentConfigurationBase):
    """A Decision Agent as returned by the API."""

    id: Annotated[UUID, Field(description="ID of the Agent.")]
