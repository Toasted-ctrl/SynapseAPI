from pydantic import BaseModel
from uuid import UUID


class MCP(BaseModel):
    id: UUID
    name: str


class McpsRead(BaseModel):
    mcps: list[MCP]
