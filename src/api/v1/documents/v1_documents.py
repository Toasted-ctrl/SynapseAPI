import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from database.documents_users import (
    delete_user_document_by_document_id,
    get_user_documents_by_scope,
)
from database.session import get_db_session

from .models import DocumentDeleteRead, DocumentsRead

router = APIRouter()


@router.get(
    "/documents",
    tags=["Documents"],
    description="Retrieve a list of stored documents by scope.",
    response_model=DocumentsRead,
)
async def get_documents(
    scope: str,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentsRead:
    if scope in ["user_vs_files", "user_vs_memories", "user_vs_skills"]:
        return DocumentsRead(
            documents_scope=scope,
            documents=await get_user_documents_by_scope(
                session=session, user_id=user.id, scope=scope
            ),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unsupported scope: '{scope}'",
        )


@router.delete(
    "/documents",
    tags=["Documents"],
    description="Delete a document by document_id and scope.",
    response_model=DocumentDeleteRead,
)
async def delete_document(
    document_id: str,
    scope: str,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentDeleteRead:
    try:
        if scope in ["user_vs_files", "user_vs_memories", "user_vs_skills"]:
            user_document_id = uuid.UUID(document_id)
            docid = await delete_user_document_by_document_id(
                session=session, user_id=user.id, document_id=user_document_id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unsupported scope: '{scope}'",
            )
        return DocumentDeleteRead(id=docid, is_deleted=True)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Failed to delete document: {e!s}",
        )
