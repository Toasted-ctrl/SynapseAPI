from pydantic import BaseModel


class StatusRead(BaseModel):
    status: str
