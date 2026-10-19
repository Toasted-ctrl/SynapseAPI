from datetime import datetime
from pydantic import BaseModel, ConfigDict
from uuid import UUID


class ProviderModelsRead(BaseModel):
    providers: dict[str, dict[str, list[str]]]


class ModelSamplingParameters(BaseModel):
    temperature: bool
    top_k: bool
    top_p: bool


class ModelSamplingRead(BaseModel):
    provider: str
    model: str
    sampling: ModelSamplingParameters


class ProviderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    internal: bool
    requires_api_key: bool
    api_key_configured: bool | None = None
    api_key_id: UUID | None = None
    api_key_short: str | None = None
    api_key_expiry_date: datetime | None = None


class ProvidersRead(BaseModel):
    providers: list[ProviderRead]