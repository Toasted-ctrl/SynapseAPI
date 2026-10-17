from datetime import datetime
from pydantic import BaseModel
from uuid import UUID


class UserKeyCreate(BaseModel):
    provider: str
    api_key: str


class UserKeyRead(BaseModel):
    user_id: UUID
    provider: str
    api_key_short: str
    expires: datetime
