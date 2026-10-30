import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, status

from api.v1.inference.helpers import (
    ConversationSettings,
    ModelType,
    VectorEmbeddingSettings,
    get_provider_configuration,
)
from api.v1.inference.models import VectorEmbeddingModel
from api.v1.playground.models import AgentModel, AgentTools
from database.providers import ProviderConfiguration, UserProviderRegistry
from providers.dataclasses import LangChainCon

pytestmark = pytest.mark.asyncio

MODELS = {
    "chat_completion": ["llama3", "gpt-5"],
    "vector_embedding": ["nomic-embed-text"],
    "translation": ["translategemma"],
}


def _provider(name: str, requires_api_key: bool, api_key_configured: bool):
    return ProviderConfiguration(
        id=uuid.uuid4(),
        name=name,
        base_url=f"http://{name}.local",
        langchain_con=LangChainCon.OLLAMA,
        api_key_configured=api_key_configured,
        internal=not requires_api_key,
        requires_api_key=requires_api_key,
    )


@pytest.fixture
def registry() -> UserProviderRegistry:
    return UserProviderRegistry(
        [
            _provider("ollama", requires_api_key=False, api_key_configured=False),
            _provider("openai", requires_api_key=True, api_key_configured=True),
            _provider("anthropic", requires_api_key=True, api_key_configured=False),
        ]
    )


@pytest.fixture
def models(monkeypatch):
    """Patches model_config so each model type returns its list from MODELS."""
    mocks = {
        model_type: AsyncMock(return_value=list(names))
        for model_type, names in MODELS.items()
    }
    for model_type, mock in mocks.items():
        monkeypatch.setattr(
            f"api.v1.inference.helpers.model_config.{model_type}_models", mock
        )
    return mocks


def _settings(provider: str, model: str) -> ConversationSettings:
    return ConversationSettings(
        model=AgentModel(provider=provider, name=model),
        system_prompt=None,
        tools=AgentTools(),
    )


@pytest.mark.parametrize("provider", ["ollama", "openai"])
async def test_returns_configuration_for_usable_provider(registry, models, provider):
    prov = await get_provider_configuration(
        user_provider_registry=registry,
        settings=_settings(provider, "llama3"),
        model_type=ModelType.CHAT_COMPLETION,
    )
    assert prov is registry[provider]


async def test_unknown_provider_returns_404(registry, models):
    with pytest.raises(HTTPException) as exc:
        await get_provider_configuration(
            user_provider_registry=registry,
            settings=_settings("mistral", "llama3"),
            model_type=ModelType.CHAT_COMPLETION,
        )
    assert exc.value.status_code == status.HTTP_404_NOT_FOUND
    assert "mistral" in exc.value.detail
    for mock in models.values():
        mock.assert_not_awaited()


async def test_provider_missing_api_key_returns_400(registry, models):
    with pytest.raises(HTTPException) as exc:
        await get_provider_configuration(
            user_provider_registry=registry,
            settings=_settings("anthropic", "llama3"),
            model_type=ModelType.CHAT_COMPLETION,
        )
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "anthropic" in exc.value.detail
    for mock in models.values():
        mock.assert_not_awaited()


async def test_accepts_vector_embedding_settings(registry, models):
    prov = await get_provider_configuration(
        user_provider_registry=registry,
        settings=VectorEmbeddingSettings(
            model=VectorEmbeddingModel(provider="ollama", name="nomic-embed-text")
        ),
        model_type=ModelType.VECTOR_EMBEDDING,
    )
    assert prov is registry["ollama"]
    models["vector_embedding"].assert_awaited_once()


@pytest.mark.parametrize(
    ("model_type", "model"),
    [
        (ModelType.CHAT_COMPLETION, "gpt-5"),
        (ModelType.VECTOR_EMBEDDING, "nomic-embed-text"),
        (ModelType.TRANSLATION, "translategemma"),
    ],
)
async def test_checks_model_against_matching_type(registry, models, model_type, model):
    await get_provider_configuration(
        user_provider_registry=registry,
        settings=_settings("openai", model),
        model_type=model_type,
    )
    for name, mock in models.items():
        if name == model_type:
            mock.assert_awaited_once()
        else:
            mock.assert_not_awaited()


@pytest.mark.parametrize(
    ("model_type", "model"),
    [
        (ModelType.CHAT_COMPLETION, "nomic-embed-text"),
        (ModelType.VECTOR_EMBEDDING, "llama3"),
        (ModelType.TRANSLATION, "gpt-5"),
    ],
)
async def test_unsupported_model_returns_404(registry, models, model_type, model):
    with pytest.raises(HTTPException) as exc:
        await get_provider_configuration(
            user_provider_registry=registry,
            settings=_settings("openai", model),
            model_type=model_type,
        )
    assert exc.value.status_code == status.HTTP_404_NOT_FOUND
    assert exc.value.detail == f"Unsupported {model_type} model: {model}"
