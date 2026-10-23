from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class UserKeyCreate(BaseModel):
    provider: str
    api_key: str
    expiry_days: int | None = Field(default=None, ge=30)


class UserKeyRead(BaseModel):
    user_id: UUID
    provider: str
    api_key_short: str
    expires: datetime
