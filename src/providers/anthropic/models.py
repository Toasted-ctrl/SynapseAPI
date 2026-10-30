from anthropic import AsyncAnthropic, AuthenticationError

from core.logging import get_logger
from core.model_types import model_config
from security.encryption import decrypt

log = get_logger()


async def get_models(encrypted_api_key: str) -> dict[str, list[str]]:
    """Function that will return all available Anthropic models."""
    log.debug(f"Fetching Anthropic models with key '{encrypted_api_key[:10]}'...")
    try:
        async with AsyncAnthropic(api_key=decrypt(content=encrypted_api_key)) as client:
            models = await client.models.list()
            cc_models = await model_config.chat_completion_models()
            tm_models = await model_config.translation_models()
            ve_models = await model_config.vector_embedding_models()
            so_models = await model_config.system_one_models()
            return {
                "chat_completion": [
                    model.id for model in models.data if model.id in cc_models
                ],
                "translation": [
                    model.id for model in models.data if model.id in tm_models
                ],
                "vector_embedding": [
                    model.id for model in models.data if model.id in ve_models
                ],
                "system_one": [
                    model.id for model in models.data if model.id in so_models
                ],
            }
    except AuthenticationError as e:
        log.debug(f"Could not retrieve Anthropic models: {e}")
        return {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": [],
            "system_one": [],
        }
