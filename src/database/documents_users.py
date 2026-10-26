import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.vector_store import VectorStoreConfig, get_vector_store_settings
from vs.delete_document import delete_document_from_qdrant_by_user_document_id

from .schemas.documents_user import DocumentsUsersT
from .schemas.user_skills import UserSkillsT

log = get_logger()


async def store_user_document(
    session: AsyncSession, user_id: uuid.UUID, name: str, scope: str
) -> uuid.UUID:
    exists = (
        await session.scalars(
            select(DocumentsUsersT.id).where(
                DocumentsUsersT.name == name,
                DocumentsUsersT.scope == scope,
                DocumentsUsersT.user_id == user_id,
            )
        )
    ).all()
    if exists:
        raise ValueError(f"File '{name}' with scope '{scope}' scope already exists.")
    nf = DocumentsUsersT(user_id=user_id, name=name, scope=scope)
    session.add(nf)
    await session.flush()
    return nf.id


async def get_user_documents_by_scope(
    session: AsyncSession, user_id: uuid.UUID, scope: str
) -> list[dict]:
    docs = (
        await session.scalars(
            select(DocumentsUsersT).where(
                DocumentsUsersT.user_id == user_id, DocumentsUsersT.scope == scope
            )
        )
    ).all()
    return [{"name": doc.name, "id": doc.id} for doc in docs]


async def delete_user_document_by_document_id(
    session: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID
) -> uuid.UUID | None:
    doc = await session.scalar(
        select(DocumentsUsersT).where(
            DocumentsUsersT.user_id == user_id, DocumentsUsersT.id == document_id
        )
    )
    if doc is None:
        log.debug(f"Document '{document_id}' does not exist for User '{user_id}'")
        raise ValueError(f"Document '{document_id}' does not exist")

    docid = doc.id
    scope = doc.scope

    await session.delete(doc)
    await session.flush()

    log.debug(f"Removed User document with document ID '{document_id}' from database.")

    # TODO: Currently this only supports Qdrant, but we should implement more Vector Stores.

    if scope in ["user_vs_files", "user_vs_memories", "user_vs_skills"]:
        vscf: VectorStoreConfig = await get_vector_store_settings(
            scope=scope, session=session
        )

        await delete_document_from_qdrant_by_user_document_id(
            user_id=user_id,
            user_document_id=docid,
            collection_name=vscf.vs_collection_name,
            url=vscf.vs_base_url,
            port=vscf.vs_port,
            encrypted_api_key=vscf.vs_encrypted_api_key,
        )

        # Only delete skill if this is a user_vs_skills document
        if scope == "user_vs_skills":
            skill = await session.scalar(
                select(UserSkillsT).where(
                    UserSkillsT.id == docid, UserSkillsT.user_id == user_id
                )
            )

            if skill:
                await session.delete(skill)
                await session.flush()

    return docid
