from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class MCP(BaseModel):
    id: UUID
    name: str


class McpsRead(BaseModel):
    mcps: list[MCP]


class MCPConfigurationBase(BaseModel):
    name: Annotated[str, Field(description="Name of the MCP.")]
    transport: Annotated[
        Literal["streamable_http"],
        Field(
            description="Transport Type of the MCP. Currently only streamable_http is supported."
        ),
    ] = "streamable_http"
    url: Annotated[str, Field(description="URL of the MCP.")]


class MCPConfigurationRead(MCPConfigurationBase):
    id: Annotated[UUID, Field(description="ID of the MCP.")]


class MCPConfigurationCreate(MCPConfigurationBase):
    pass
