from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_admin import depends_verify_admin
from core.model_types import model_config
from database.models import ModelAlreadyExistsError, add_model_with_expertise
from database.session import get_db_session

from .models import ModelCreate, ModelRead

router = APIRouter()


@router.post(
    "/admin/models",
    tags=["Admin / Models"],
    description="Register a new model with an expertise. Requires Admin permissions.",
    response_model=ModelRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(depends_verify_admin)],
)
async def admin_add_model_with_expertise(
    model: ModelCreate,
    session: AsyncSession = Depends(get_db_session),
) -> ModelRead:
    try:
        created = await add_model_with_expertise(
            session=session, model_name=model.model_name, expertise=model.expertise
        )
    except ModelAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    model_config.invalidate(model.expertise)
    return ModelRead(**asdict(created))
