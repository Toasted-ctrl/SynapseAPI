from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from cachetools import TTLCache

from core.model_types import ModelConfig

pytestmark = pytest.mark.asyncio

MODELS = {
    "chat_completion": ["llama3", "gpt-5"],
    "vector_embedding": ["nomic-embed-text"],
    "translation": ["translategemma"],
    "system_one": ["gpt-5"],
}


@pytest.fixture
def db(monkeypatch):
    """Patches the DB layer; returns (session, get_models_by_expertise mock)."""
    session = MagicMock()

    @asynccontextmanager
    async def fake_ctx():
        yield session

    get_models = AsyncMock(
        side_effect=lambda session, expertise: list(MODELS[expertise])
    )
    monkeypatch.setattr("core.model_types.get_db_session_ctx", fake_ctx)
    monkeypatch.setattr("core.model_types.get_models_by_expertise", get_models)
    return session, get_models


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TestModelConfig:
    @pytest.mark.parametrize(
        "method, expertise",
        [
            ("chat_completion_models", "chat_completion"),
            ("vector_embedding_models", "vector_embedding"),
            ("translation_models", "translation"),
            ("system_one_models", "system_one"),
        ],
    )
    async def test_returns_models_for_expertise(self, db, method, expertise):
        session, get_models = db

        result = await getattr(ModelConfig(), method)()

        assert result == MODELS[expertise]
        get_models.assert_awaited_once_with(session=session, expertise=expertise)

    async def test_second_call_is_served_from_cache(self, db):
        _, get_models = db
        mc = ModelConfig()

        first = await mc.chat_completion_models()
        second = await mc.chat_completion_models()

        assert first == second == MODELS["chat_completion"]
        get_models.assert_awaited_once()

    async def test_each_expertise_is_cached_separately(self, db):
        _, get_models = db
        mc = ModelConfig()

        await mc.chat_completion_models()
        await mc.translation_models()
        await mc.chat_completion_models()
        await mc.translation_models()

        assert [c.kwargs["expertise"] for c in get_models.await_args_list] == [
            "chat_completion",
            "translation",
        ]

    async def test_refetches_after_ttl_expires(self, db):
        _, get_models = db
        clock = FakeClock()
        mc = ModelConfig()
        mc._cache = TTLCache(maxsize=10, ttl=600, timer=clock)

        await mc.chat_completion_models()
        clock.now = 599
        await mc.chat_completion_models()
        assert get_models.await_count == 1

        clock.now = 601
        await mc.chat_completion_models()
        assert get_models.await_count == 2

    async def test_default_ttl_is_ten_minutes(self):
        assert ModelConfig()._cache.ttl == 600

    async def test_empty_result_is_not_cached(self, db):
        _, get_models = db
        get_models.side_effect = None
        get_models.return_value = []
        mc = ModelConfig()

        assert await mc.translation_models() == []
        assert await mc.translation_models() == []
        assert get_models.await_count == 2

    async def test_models_seeded_after_empty_result_are_picked_up_and_cached(self, db):
        _, get_models = db
        get_models.side_effect = [[], ["translategemma"], ["should-not-be-fetched"]]
        mc = ModelConfig()

        assert await mc.translation_models() == []
        assert await mc.translation_models() == ["translategemma"]
        assert await mc.translation_models() == ["translategemma"]
        assert get_models.await_count == 2

    async def test_failure_is_not_cached(self, db):
        _, get_models = db
        get_models.side_effect = [RuntimeError("db down"), ["llama3"]]
        mc = ModelConfig()

        with pytest.raises(RuntimeError, match="db down"):
            await mc.chat_completion_models()

        assert await mc.chat_completion_models() == ["llama3"]

    async def test_instances_do_not_share_cache(self, db):
        _, get_models = db

        await ModelConfig().chat_completion_models()
        await ModelConfig().chat_completion_models()

        assert get_models.await_count == 2
