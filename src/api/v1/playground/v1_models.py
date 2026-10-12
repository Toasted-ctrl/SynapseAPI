from typing import Annotated
from uuid import UUID
from pydantic import BaseModel, Field


class AgentModelParameters(BaseModel):
    """Sampling parameters passed to the model provider."""
    temperature: Annotated[float | None, Field(ge=0.0, le=2.0, description="Controls randomness. Lower values make output more focused and deterministic; higher values make it more varied.")] = None
    top_p: Annotated[float | None, Field(gt=0.0, le=1.0, description="Nucleus sampling. The model samples only from the smallest set of tokens whose cumulative probability exceeds top_p. Lower values restrict output to more likely tokens.")] = None
    top_k: Annotated[int | None, Field(ge=1, description="Restricts sampling to the k most likely tokens at each step. Lower values make output more predictable.")] = None


class AgentModel(BaseModel):
    """The model the Agent runs on."""
    provider: Annotated[str, Field(min_length=1, description="Model provider, e.g. 'anthropic'.")]
    name: Annotated[str, Field(min_length=1, description="Provider-specific model name.")]
    parameters: Annotated[AgentModelParameters, Field(default_factory=AgentModelParameters, description="Sampling parameters for the model.")]


class AgentTools(BaseModel):
    """Tools the Agent is allowed to use."""
    mcp_ids: Annotated[list[UUID], Field(default_factory=list, description="List of MCP servers that Agent is allowed to use.")]


class AgentConfigurationBase(BaseModel):
    """Fields shared by all Agent configuration models.

    Everything except name is stored as JSON by database.agent, so changes here must
    follow the settings versioning rules described there.
    """
    name: Annotated[str, Field(min_length=1, description="Name of the Agent.")]
    model: AgentModel
    system_prompt: Annotated[str | None, Field(description="System prompt given to the Agent.")] = None
    tools: Annotated[AgentTools, Field(default_factory=AgentTools, description="Tools the Agent is allowed to use.")]


class AgentConfigurationCreate(AgentConfigurationBase):
    """Request body for creating or updating an Agent. The server assigns the id."""


class AgentConfigurationRead(AgentConfigurationBase):
    """An Agent as returned by the API."""
    id: Annotated[UUID, Field(description="ID of the Agent.")]
