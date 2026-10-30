from pydantic import BaseModel


class Contact(BaseModel):
    maintainer: str


class RootRead(BaseModel):
    version: str
    application_name: str
    contact: Contact


class StatusRead(BaseModel):
    status: str
