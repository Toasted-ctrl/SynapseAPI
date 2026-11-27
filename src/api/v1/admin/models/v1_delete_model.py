from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_admin import depends_verify_admin
from core.model_types import model_config
from database.models import ModelNotFoundError, delete_model_by_name
from database.session import get_db_session

router = APIRouter()


@router.delete(
    "/admin/models",
    tags=["Admin / Models"],
    description="Deletes a model. Requires Admin permissions.",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Model does not exist."}},
    dependencies=[Depends(depends_verify_admin)],
)
async def admin_delete_model(
    model_name: str = Query(min_length=1, max_length=255),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    try:
        deleted = await delete_model_by_name(session=session, model_name=model_name)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    model_config.invalidate(deleted.expertise)
