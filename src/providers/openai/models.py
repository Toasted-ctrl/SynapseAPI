from openai import AsyncOpenAI, AuthenticationError

from core.logging import get_logger
from core.model_types import model_config
from security.encryption import decrypt


log = get_logger()


async def get_models(
    encrypted_api_key: str,
    base_url: str
) -> dict[str, list[str]]:
    """Fetches and returns all models supported by the API Key.
    This method will work for both OpenAI as well as Melious and Z.ai endpoints."""
    log.debug(f"Fetching models for Provider URL '{base_url}' with API Key '{encrypted_api_key[:10]}'...")
    try:
        async with AsyncOpenAI(
            api_key=decrypt(content=encrypted_api_key),
            base_url=base_url
        ) as client:
            models = await client.models.list()
            cc_models = await model_config.chat_completion_models()
            tm_models = await model_config.translation_models()
            ve_models = await model_config.vector_embedding_models()
            return {
                "chat_completion": [model.id for model in models.data if model.id in cc_models],
                "translation": [model.id for model in models.data if model.id in tm_models],
                "vector_embedding": [model.id for model in models.data if model.id in ve_models]
            }
    except AuthenticationError as e:
        log.debug(f"Could not retrieve OpenAI models: {str(e)}")
        return {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": []
        }