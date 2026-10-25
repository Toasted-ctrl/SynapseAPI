from uuid import UUID

from pydantic import BaseModel


class MCP(BaseModel):
    id: UUID
    name: str


class McpsRead(BaseModel):
    mcps: list[MCP]
