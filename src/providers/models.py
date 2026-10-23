from httpx import ConnectError, ConnectTimeout
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from core.logging import get_logger
from database.providers import get_all_provider_configurations
from providers.anthropic.models import get_models as get_models_anthropic
from providers.google.models import get_models as get_models_google
from providers.ollama.models import get_models as get_models_ollama
from providers.openai.models import get_models as get_models_openai


log = get_logger()


async def get_all_models(
    session: AsyncSession,
    user_id: UUID
) -> dict:
    """Returns a dictionary of all LLM providers, separated by provider.
    Per provider, all models are listed by area of expertise
    (e.g., chat_completion, translation, vector_embedding)."""
    p_reg = await get_all_provider_configurations(
        session=session,
        user_id=user_id
    )
    all_m = {}
    for p in p_reg:
        if p.requires_api_key and not p.api_key_configured:
            log.debug(f"Provider '{p.name}' not configured for User '{user_id}'...")
            continue
        try:
            if p.langchain_con == "ChatOllama":
                mds = await get_models_ollama(
                    base_url=p.base_url
                )
            # Anthropic based connectors
            elif p.langchain_con == "ChatAnthropic":
                mds = await get_models_anthropic(
                    encrypted_api_key=p.encrypted_api_key
                )
            # Google based connectors
            elif p.langchain_con == "ChatGoogleGenerativeAI":
                mds = await get_models_google(
                    encrypted_api_key=p.encrypted_api_key
                )
            # OpenAI based connectors
            elif p.langchain_con == "ChatOpenAI":
                mds = await get_models_openai(
                    encrypted_api_key=p.encrypted_api_key,
                    base_url=p.base_url
                )
            else:
                continue
            all_m[p.name] = mds
        except (ConnectTimeout, ConnectError):
            continue
    return all_m