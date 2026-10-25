from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AgentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str


class AgentListRead(BaseModel):
    agents: list[AgentListItem]
