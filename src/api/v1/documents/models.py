from uuid import UUID

from pydantic import BaseModel


class Document(BaseModel):
    name: str
    id: UUID


class DocumentsRead(BaseModel):
    documents_scope: str
    documents: list[Document]


class DocumentDeleteRead(BaseModel):
    id: UUID
    is_deleted: bool


class Metadata(BaseModel):
    document_name: str


class VectorStoreDocumentsCreate(BaseModel):
    texts: list[str]
    metadatas: list[Metadata]


class VectorStoreDocumentsRead(BaseModel):
    added_documents: list[UUID]


class VectorStoreSearchCreate(BaseModel):
    query: str


class SkillCreate(BaseModel):
    name: str
    description: str
    skill_text: str
    parameters_schema: dict | None = None


class SkillCreateRead(BaseModel):
    skill_id: UUID
    name: str
    message: str


class SkillRead(BaseModel):
    skill_id: UUID
    name: str
    description: str | None = None
    instructions: str
    parameter_schema: dict | None = None
    is_encrypted: bool = False


class SkillReplace(BaseModel):
    """Complete replacement for a skill (PUT). All fields are required and always encrypted."""

    skill_id: UUID
    name: str
    description: str
    skill_text: str
    parameters_schema: dict | None = None
