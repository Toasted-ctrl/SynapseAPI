from fastapi import APIRouter, Depends
from fastapi_cache.decorator import cache
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.cache import cache_key_builder
from database.providers import get_all_provider_configurations
from database.session import get_db_session

from .models import ProviderRead, ProvidersRead

router = APIRouter()


@router.get(
    "/providers/configuration",
    response_model=ProvidersRead,
    description=(
        "Returns a list of Providers that are available through the API. "
        "Provides an indication of whether a User-provided Provider API key is required, "
        "and if a Provider API key has been configured."
    ),
    tags=["Providers"],
)
@cache(expire=600, key_builder=cache_key_builder)
async def get_all_providers(
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> ProvidersRead:
    p_reg = await get_all_provider_configurations(session=session, user_id=user.id)
    return ProvidersRead(providers=[ProviderRead.model_validate(p) for p in p_reg])
