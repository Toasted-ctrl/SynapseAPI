from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_admin import depends_verify_admin
from core.model_types import ModelType
from database.models import get_models_by_expertise
from database.session import get_db_session

from .models import ModelsByExpertiseRead

router = APIRouter()


@router.get(
    "/admin/models",
    tags=["Admin / Models"],
    description="Get a list of models by expertise. Requires Admin permissions.",
    response_model=ModelsByExpertiseRead,
    dependencies=[Depends(depends_verify_admin)],
)
async def admin_get_models_by_expertise(
    expertise: ModelType = ModelType.CHAT_COMPLETION,
    session: AsyncSession = Depends(get_db_session),
) -> ModelsByExpertiseRead:
    models = await get_models_by_expertise(session=session, expertise=expertise)
    return ModelsByExpertiseRead(expertise=expertise, models=models)
