from dataclasses import asdict
from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi_cache.decorator import cache
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import depends_verify_user, VerifiedUser
from core.cache import cache_key_builder
from core.logging import get_logger
from database.model_sampling import get_model_sampling_settings, ModelSampling
from database.session import get_db_session
from providers.models import get_all_models

from .models import ProviderModelsRead, ModelSamplingRead


log = get_logger()


router = APIRouter()


@router.get(
    "/providers/models",
    tags=["Providers"],
    response_model=ProviderModelsRead,
    description="Returns a list of available models, divided by Provider and model expertise."
)
@cache(expire=1800, key_builder=cache_key_builder)
async def get_models(
    request: Request,
    response: Response,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session)
) -> ProviderModelsRead:
    pm = await get_all_models(
        session=session,
        user_id=user.id
    )
    if not pm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models available"
        )
    return ProviderModelsRead(
        providers=pm
    )


@router.get(
    "/providers/models/sampling",
    tags=["Providers"],
    description="Returns a dictionary of support model sampling settings.",
    response_model=ModelSamplingRead
)
@cache(expire=1800, key_builder=cache_key_builder)
async def get_model_sampling(
    request: Request,
    response: Response,
    provider_name: str,
    model_name: str,
    session: AsyncSession = Depends(get_db_session)
) -> ModelSamplingRead:
    sampling: ModelSampling = await get_model_sampling_settings(
        session=session,
        provider_name=provider_name,
        model_name=model_name
    )
    return ModelSamplingRead(
        provider=provider_name,
        model=model_name,
        sampling=asdict(sampling)
    )