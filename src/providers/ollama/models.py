from ollama import AsyncClient

from core.logging import get_logger
from core.model_types import model_config

log = get_logger()


async def get_models(base_url: str) -> dict[str, list[str]]:
    """Fetches and returns a dictionary of available models for the Ollama instance,
    subdivided by model 'Expertise'.
    Will return a dictionary with empty lists, if a connection error with the Ollama instance would occur."""

    try:
        async with AsyncClient(host=base_url) as client:
            response = await client.list()
    except ConnectionError:
        log.warning(f"Ollama Provider at {base_url} is unreachable, skipping ...")
        return {"chat_completion": [], "translation": [], "vector_embedding": []}

    mds = [model.model for model in response.models]
    cc_models = await model_config.chat_completion_models()
    ve_models = await model_config.vector_embedding_models()
    tm_models = await model_config.translation_models()

    # Chat Completion models
    cc = [m for m in mds if m in cc_models]

    # Vector Embedding models
    ve = [m for m in mds if m in ve_models]

    # Translation models
    tm = [m for m in mds if m in tm_models]

    return {"chat_completion": cc, "translation": tm, "vector_embedding": ve}
