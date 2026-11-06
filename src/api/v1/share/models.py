from uuid import UUID

from pydantic import BaseModel


class UserAgentAccessRead(BaseModel):
    users: list[UUID]


class GrantAgentAccessRead(BaseModel):
    id: UUID
