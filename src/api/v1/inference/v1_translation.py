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

router = APIRouter(tags=["Inference / Translation"])


@router.post(
    "/inference/translation/translategemma",
    description="Invokes a translation from translategemma.",
    response_model=TranslationRead,
)
async def post_translation_translategemma(
    payload: TranslationCreate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> TranslationRead:
    try:
        unsupported = {payload.from_lang_code, payload.to_lang_code}.difference(
            translategemma_languages()
        )
        if unsupported:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language: {', '.join(sorted(unsupported))}",
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
        return TranslationRead(
            **get_translation_translategemma(
                from_lang=payload.from_lang_code,
                to_lang=payload.to_lang_code,
                prompt=payload.prompt,
                temperature=payload.parameters.temperature,
                base_url=provider.base_url,
            )
        )
    except ConnectTimeout, ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Provider",
        )


@router.get(
    "/inference/translation",
    response_model=TranslationLanguagesRead,
    description="Returns languages supported by the indicated model.",
)
@cache(expire=300, key_builder=cache_key_builder)
async def get_supported_languages(
    request: Request,
    response: Response,
    model_name: str = "translategemma",
    user: VerifiedUser = Depends(depends_verify_user),
) -> TranslationLanguagesRead:
    if model_name != "translategemma":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unsupported translation model: {model_name}",
        )
    return TranslationLanguagesRead(languages=translategemma_languages())
