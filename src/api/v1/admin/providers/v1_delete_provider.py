from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_admin import depends_verify_admin
from database.providers import ProviderNotFoundError, delete_provider_record_by_id
from database.session import get_db_session

router = APIRouter()


@router.delete(
    "/admin/providers",
    tags=["Admin / Providers"],
    description="Delete a provider by ID. Requires Admin permissions.",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Provider does not exist."},
    },
    dependencies=[Depends(depends_verify_admin)],
)
async def admin_delete_provider(
    id: UUID = Query(description="The UUID of the provider to delete."),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    """Delete a provider from the database by its ID.

    Args:
        id: The UUID of the provider to delete.
        session: The async SQLAlchemy session.

    Raises:
        HTTPException 404: If no provider with the given ID exists.
    """
    try:
        await delete_provider_record_by_id(id=id, session=session)
    except ProviderNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
