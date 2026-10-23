from pydantic import BaseModel
from uuid import UUID


class AuthenticatedUserRead(BaseModel):
    user_id: UUID
    first_name: str
    last_name: str
    email: str
