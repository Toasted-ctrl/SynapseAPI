from pydantic import BaseModel
from typing import Optional
from uuid import UUID


# Document models
class Document(BaseModel):
    name: str
    id: UUID


class DocumentsRead(BaseModel):
    documents_scope: str
    documents: list[Document]


class DocumentDeleteRead(BaseModel):
    id: UUID
    is_deleted: bool


# Vector Embedding models
class VectorEmbeddingCreate(BaseModel):
    prompt: str
    provider: str
    model: str
    dimensions: int | None = None


class VectorEmbeddingRead(BaseModel):
    prompt: str
    provider: str
    model: str
    dimensions: int
    embedding: list[float]


# Vector Store models
class Metadata(BaseModel):
    document_name: str


class VectorStoreDocumentsCreate(BaseModel):
    texts: list[str]
    metadatas: list[Metadata]


class VectorStoreDocumentsRead(BaseModel):
    added_documents: list[UUID]


class VectorStoreSearchCreate(BaseModel):
    query: str


# Skill models
class SkillCreate(BaseModel):
    name: str
    description: str
    skill_text: str
    parameters_schema: Optional[dict] = None


class SkillCreateRead(BaseModel):
    skill_id: UUID
    name: str
    message: str


class SkillRead(BaseModel):
    skill_id: UUID
    name: str
    description: str
    instructions: str
    parameter_schema: Optional[dict] = None


class SkillUpdate(BaseModel):
    skill_id: UUID
    instructions: str
