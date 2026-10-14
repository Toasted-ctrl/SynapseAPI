from pydantic import BaseModel, ConfigDict
from uuid import UUID


class AgentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str


class AgentListRead(BaseModel):
    agents: list[AgentListItem]