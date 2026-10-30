from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi_cache.decorator import cache
from httpx import ConnectError, ConnectTimeout
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.cache import cache_key_builder
from database.providers import (
    ProviderConfiguration,
    UserProviderRegistry,
    get_all_provider_configurations,
)
from database.session import get_db_session
from providers.dataclasses import LangChainCon
from providers.ollama.translategemma import (
    get_translation_translategemma,
    translategemma_languages,
)

from .helpers import ModelType, TranslationSettings, get_provider_configuration
from .models import TranslationCreate, TranslationLanguagesRead, TranslationRead

router = APIRouter(tags=["Inference / Translation"])


def _is_translategemma(model_name: str) -> bool:
    """Translategemma is currently the only translation model with an implementation."""
    return model_name.split(":")[0] == "translategemma"


@router.post(
    "/inference/translation",
    description="Translates the prompt with the requested translation model.",
    response_model=TranslationRead,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unsupported language, or the provider is not configured or does not support the model."
        },
        status.HTTP_404_NOT_FOUND: {"description": "Provider or model does not exist."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Unable to connect to provider."
        },
    },
)
async def get_translation(
    payload: TranslationCreate,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> TranslationRead:
    upr: UserProviderRegistry = await get_all_provider_configurations(
        session=session, user_id=user.id
    )

    prov: ProviderConfiguration = await get_provider_configuration(
        user_provider_registry=upr,
        settings=TranslationSettings(model=payload.model),
        model_type=ModelType.TRANSLATION,
    )

    if (
        not _is_translategemma(payload.model.name)
        or prov.langchain_con != LangChainCon.OLLAMA
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Translation with '{payload.model.name}' is not supported by provider: {prov.name}",
        )

    unsupported = {payload.from_lang_code, payload.to_lang_code}.difference(
        translategemma_languages()
    )
    if unsupported:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {', '.join(sorted(unsupported))}",
        )

    try:
        translated = await get_translation_translategemma(
            model=payload.model.name,
            from_lang=payload.from_lang_code,
            to_lang=payload.to_lang_code,
            prompt=payload.prompt,
            temperature=payload.parameters.temperature,
            base_url=prov.base_url,
        )
    except ConnectTimeout, ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Provider",
        )

    return TranslationRead(
        model=payload.model,
        prompt=payload.prompt,
        translation=translated,
        from_lang_code=payload.from_lang_code,
        to_lang_code=payload.to_lang_code,
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
    model_name: str = "translategemma:latest",
    user: VerifiedUser = Depends(depends_verify_user),
) -> TranslationLanguagesRead:
    if not _is_translategemma(model_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unsupported translation model: {model_name}",
        )
    return TranslationLanguagesRead(languages=translategemma_languages())
