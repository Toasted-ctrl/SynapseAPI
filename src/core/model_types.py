from enum import StrEnum

from cachetools import TTLCache

from database.models import get_models_by_expertise
from database.session import get_db_session_ctx


class ModelType(StrEnum):
    """Model expertise categories, as stored in the `models` table."""

    CHAT_COMPLETION = "chat_completion"
    VECTOR_EMBEDDING = "vector_embedding"
    TRANSLATION = "translation"
    SYSTEM_ONE = "system_one"


class ModelConfig:
    def __init__(self) -> None:
        self._cache: TTLCache[str, list[str]] = TTLCache(maxsize=10, ttl=600)

    async def _get_models(self, expertise: ModelType) -> list[str]:
        if expertise in self._cache:
            return self._cache[expertise]

        async with get_db_session_ctx() as session:
            models = await get_models_by_expertise(session=session, expertise=expertise)

        # Not cached when empty, so newly seeded models are picked up immediately.
        if models:
            self._cache[expertise] = models
        return models

    def invalidate(self, expertise: str) -> None:
        """Drops the cached models for an expertise in this process."""
        self._cache.pop(expertise, None)

    async def chat_completion_models(self) -> list[str]:
        return await self._get_models(ModelType.CHAT_COMPLETION)

    async def vector_embedding_models(self) -> list[str]:
        return await self._get_models(ModelType.VECTOR_EMBEDDING)

    async def translation_models(self) -> list[str]:
        return await self._get_models(ModelType.TRANSLATION)

    async def system_one_models(self) -> list[str]:
        return await self._get_models(ModelType.SYSTEM_ONE)


model_config = ModelConfig()
