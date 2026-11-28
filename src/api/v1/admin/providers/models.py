from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from providers.dataclasses import LangChainCon


class ProviderCreate(BaseModel):
    """Request model for creating a new provider."""

    name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=50,
            description="Name of the Provider (e.g., 'Ollama', 'OpenAI').",
        ),
    ]
    base_url: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description="Base URL of the Provider's API endpoint.",
        ),
    ]
    internal: Annotated[
        bool,
        Field(
            description="Whether the Provider is internal (True) or external (False).",
        ),
    ]
    requires_api_key: Annotated[
        bool,
        Field(
            description="Whether the Provider requires an API key for authentication.",
        ),
    ]
    langchain_con: Annotated[
        LangChainCon,
        Field(
            description="The LangChain connection type to use with this Provider.",
        ),
    ]
    mac_address: Annotated[
        str | None,
        Field(
            max_length=30,
            default=None,
            description="Optional MAC address associated with the Provider.",
        ),
    ] = None


class ProviderRead(BaseModel):
    """Response model for a provider record."""

    id: Annotated[
        UUID,
        Field(description="Unique identifier for the Provider."),
    ]
    name: Annotated[
        str,
        Field(description="Name of the Provider."),
    ]
    base_url: Annotated[
        str,
        Field(description="Base URL of the Provider's API endpoint."),
    ]
    internal: Annotated[
        bool,
        Field(description="Whether the Provider is internal."),
    ]
    requires_api_key: Annotated[
        bool,
        Field(description="Whether the Provider requires an API key."),
    ]
    langchain_con: Annotated[
        LangChainCon,
        Field(description="The LangChain connection type."),
    ]
    mac_address: Annotated[
        str | None,
        Field(description="Optional MAC address associated with the Provider."),
    ] = None
    host: Annotated[
        str | None,
        Field(description="Optional host associated with the Provider."),
    ] = None


class ProviderListRead(BaseModel):
    """Response model for a list of provider records."""

    providers: Annotated[
        list[ProviderRead],
        Field(description="List of all Provider records."),
    ]
