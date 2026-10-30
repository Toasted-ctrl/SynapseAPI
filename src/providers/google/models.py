from google import genai
from google.genai.errors import ClientError

from core.logging import get_logger
from core.model_types import model_config
from security.encryption import decrypt

log = get_logger()


async def get_models(encrypted_api_key: str) -> dict[str, list[str]]:
    """Function that will return all available Google Gemini models."""
    log.debug(f"Fetching Google models with key '{encrypted_api_key[:10]}'...")
    try:
        async with genai.Client(
            api_key=decrypt(content=encrypted_api_key)
        ).aio as client:
            models = [
                model.name.removeprefix("models/")
                async for model in await client.models.list()
                if model.name
            ]
            cc_models = await model_config.chat_completion_models()
            tm_models = await model_config.translation_models()
            ve_models = await model_config.vector_embedding_models()
            so_models = await model_config.system_one_models()
            return {
                "chat_completion": [model for model in models if model in cc_models],
                "translation": [model for model in models if model in tm_models],
                "vector_embedding": [model for model in models if model in ve_models],
                "system_one": [model for model in models if model in so_models],
            }
    except ClientError as e:
        log.debug(f"Could not retrieve Google models: {e}")
        return {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": [],
            "system_one": [],
        }
