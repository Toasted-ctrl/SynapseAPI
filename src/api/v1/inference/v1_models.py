from typing import Annotated
from uuid import UUID
from pydantic import BaseModel, Field, model_validator

from api.v1.playground.v1_models import AgentModel, AgentTools


class ConversationCreate(BaseModel):
    """Request body for a conversation turn.

    Either references a stored Agent by agent_id, or configures the model and tools
    inline. The inline fields reuse the Agent configuration models, so both paths
    behave the same. A system prompt is only used when it comes from a stored Agent.
    """
    agent_id: Annotated[UUID | None, Field(description="ID of a stored Agent to converse with. Excludes model and tools.")] = None
    thread_id: Annotated[UUID | None, Field(description="ID of an existing thread to continue. A new thread is started if omitted.")] = None
    prompt: Annotated[str, Field(description="The user's message.")]

    model: Annotated[AgentModel | None, Field(description="The model to converse with. Required when agent_id is omitted.")] = None
    tools: Annotated[AgentTools | None, Field(description="Tools the model is allowed to use.")] = None

    @model_validator(mode='after')
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
    thread_id: Annotated[UUID, Field(description="ID of the thread the turn belongs to.")]
    messages: Annotated[list, Field(description="Messages in the thread.")]
