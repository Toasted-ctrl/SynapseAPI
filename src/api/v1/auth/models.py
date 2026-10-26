from uuid import UUID

from pydantic import BaseModel


class AuthenticatedUserRead(BaseModel):
    user_id: UUID
    first_name: str
    last_name: str
    email: str
