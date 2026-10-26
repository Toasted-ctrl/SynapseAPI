from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi_cache.decorator import cache
from httpx import ConnectError, ConnectTimeout
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.cache import cache_key_builder
from database.providers import ProviderConfiguration
from database.session import get_db_session
from providers.models import get_all_provider_configurations
from providers.ollama.translategemma import (
    get_translation_translategemma,
    translategemma_languages,
)

from .models import TranslationCreate, TranslationLanguagesRead, TranslationRead

router = APIRouter(tags=["Translation"])


@router.post(
    "/translation/translategemma",
    description="Invokes a translation from translategemma.",
    response_model=TranslationRead,
)
async def post_translation_translategemma(
    payload: TranslationCreate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> TranslationRead:
    try:
        if (
            not {payload.to_lang_code, payload.from_lang_code}
            <= translategemma_languages()
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language: {payload.from_lang_code}",
            )
        if payload.prompt == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt must not be None",
            )
        p = "Ollama-1"

        # Default provider for now? Perhaps we can fetch all providers by
        # Langchain con instead? Cache the result?
        # TODO: Find a way to pinpoint the correct provider.

        pr = await get_all_provider_configurations(session=session, user_id=user.id)
        provider: ProviderConfiguration = getattr(pr, p)
        return get_translation_translategemma(
            from_lang=payload.from_lang_code,
            to_lang=payload.to_lang_code,
            prompt=payload.prompt,
            temperature=payload.parameters.temperature,
            base_url=provider.base_url,
        )
    except ConnectTimeout, ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Provider",
        )


@router.get(
    "/translation/translategemma",
    response_model=TranslationLanguagesRead,
    description="Returns languages supported by translategemma.",
)
@cache(expire=300, key_builder=cache_key_builder)
async def get_languages_translategemma(
    request: Request,
    response: Response,
    user: VerifiedUser = Depends(depends_verify_user),
) -> TranslationLanguagesRead:
    return TranslationLanguagesRead(languages=translategemma_languages())
