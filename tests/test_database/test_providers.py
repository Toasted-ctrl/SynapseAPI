import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from database.providers import (
    Provider,
    ProviderConfiguration,
    UserProviderRegistry,
    get_all_provider_configurations,
    get_or_create_provider,
    get_provider_config,
    get_providers_by_id,
    get_providers_by_location,
)
from database.schemas.providers import ProvidersT
from database.user_keys import ProviderAPIKey
from db_helpers import make_session, params, row


def _provider_row(name="Ollama-1", internal=True, requires_api_key=False, **extra):
    return row(
        id=uuid.uuid4(),
        name=name,
        base_url=f"http://{name.lower()}",
        internal=internal,
        requires_api_key=requires_api_key,
        langchain_con="ChatOllama" if internal else "ChatOpenAI",
        mac_address=None,
        **extra
    )


def _config(name, internal=False, api_key_configured=False):
    return ProviderConfiguration(
        id=uuid.uuid4(),
        name=name,
        base_url=f"http://{name}",
        langchain_con="ChatOpenAI",
        api_key_configured=api_key_configured,
        internal=internal,
        requires_api_key=not internal
    )


def _key(provider_id, api_key="enc:sk", api_key_id=None):
    return ProviderAPIKey(
        user_id=uuid.uuid4(),
        api_key_id=api_key_id or uuid.uuid4(),
        api_key=api_key,
        api_key_short="sk-1234567",
        provider_id=provider_id,
        expiration_date=None
    )


class TestUserProviderRegistry:

    def setup_method(self):
        self.ollama = _config("Ollama", internal=True)
        self.openai = _config("OpenAI", api_key_configured=True)
        self.anthropic = _config("Anthropic", api_key_configured=False)
        self.reg = UserProviderRegistry([self.ollama, self.openai, self.anthropic])

    def test_attribute_and_item_access(self):
        assert self.reg.OpenAI is self.openai
        assert self.reg["Ollama"] is self.ollama

    def test_missing_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            self.reg.Mistral

    def test_missing_item_raises_key_error(self):
        with pytest.raises(KeyError):
            self.reg["Mistral"]

    def test_contains_and_iter(self):
        assert "OpenAI" in self.reg
        assert "Mistral" not in self.reg
        assert list(self.reg) == [self.ollama, self.openai, self.anthropic]

    def test_names(self):
        assert self.reg.names == ["Ollama", "OpenAI", "Anthropic"]

    def test_not_configured_lists_only_external_providers_missing_keys(self):
        assert self.reg.not_configured == ["Anthropic"]


@pytest.mark.asyncio
class TestGetOrCreateProvider:

    KWARGS = dict(
        name="Ollama-1",
        langchain_con="ChatOllama",
        base_url="http://ollama",
        internal=True,
        requires_api_key=False,
        host="ollama-host",
        mac_address="00:11:22:33:44:55"
    )

    async def test_returns_existing_provider_by_base_url(self):
        existing = _provider_row()
        session = make_session(scalar=[existing])

        result = await get_or_create_provider(session=session, **self.KWARGS)

        assert result.id == existing.id
        assert params(session.scalar.await_args)[0] == "http://ollama"
        session.add.assert_not_called()

    async def test_creates_new_provider(self):
        session = make_session(scalar=[None])

        result = await get_or_create_provider(session=session, **self.KWARGS)

        [added] = session.added
        assert isinstance(added, ProvidersT)
        assert added.host == "ollama-host"
        session.flush.assert_awaited_once()
        assert result == Provider(
            id=added.id,
            name="Ollama-1",
            base_url="http://ollama",
            internal=True,
            requires_api_key=False,
            langchain_con="ChatOllama",
            mac_address="00:11:22:33:44:55"
        )


@pytest.mark.asyncio
class TestGetAllProviderConfigurations:

    async def test_builds_registry_with_user_keys(self, monkeypatch):
        user_id = uuid.uuid4()
        ollama = _provider_row("Ollama", internal=True)
        openai = _provider_row("OpenAI", internal=False)
        anthropic = _provider_row("Anthropic", internal=False)
        session = make_session(execute=[[ollama, openai, anthropic]])
        openai_key_id = uuid.uuid4()
        keys = AsyncMock(return_value=[_key(openai.id, api_key="enc:sk-openai", api_key_id=openai_key_id)])
        monkeypatch.setattr("database.providers.get_user_active_keys", keys)

        reg = await get_all_provider_configurations(session=session, user_id=user_id)

        keys.assert_awaited_once_with(session=session, user_id=user_id)
        assert reg.names == ["Ollama", "OpenAI", "Anthropic"]
        assert reg.OpenAI.encrypted_api_key == "enc:sk-openai"
        assert reg.OpenAI.api_key_configured is True
        assert reg.OpenAI.api_key_id == openai_key_id
        assert reg.Anthropic.api_key_configured is False
        assert reg.Anthropic.encrypted_api_key is None
        assert reg.Anthropic.api_key_id is None
        assert reg.Ollama.requires_api_key is False
        assert reg.not_configured == ["Anthropic"]

    async def test_internal_provider_never_marked_configured(self, monkeypatch):
        ollama = _provider_row("Ollama", internal=True)
        session = make_session(execute=[[ollama]])
        monkeypatch.setattr("database.providers.get_user_active_keys", AsyncMock(return_value=[_key(ollama.id)]))

        reg = await get_all_provider_configurations(session=session, user_id=uuid.uuid4())

        assert reg.Ollama.api_key_configured is False
        assert reg.not_configured == []


@pytest.mark.asyncio
class TestGetProvidersByLocation:

    async def test_returns_providers(self):
        p = _provider_row()
        session = make_session(execute=[[p]])

        result = await get_providers_by_location(session=session, is_internal=True)

        assert [r.id for r in result] == [p.id]
        assert "providers.internal = true" in str(session.execute.await_args.args[0])

    async def test_returns_empty_list(self):
        session = make_session(execute=[[]])

        assert await get_providers_by_location(session=session, is_internal=False) == []


@pytest.mark.asyncio
class TestGetProvidersById:

    async def test_returns_matching_providers(self):
        a, b = _provider_row("A"), _provider_row("B")
        session = make_session(scalars=[[a, b]])

        result = await get_providers_by_id(session=session, ids=[a.id, b.id])

        assert [r.name for r in result] == ["A", "B"]
        assert params(session.scalars.await_args) == [[a.id, b.id]]

    async def test_returns_empty_list(self):
        session = make_session(scalars=[[]])

        assert await get_providers_by_id(session=session, ids=[uuid.uuid4()]) == []


@pytest.mark.asyncio
class TestGetProviderConfig:

    @pytest.fixture
    def registry(self, monkeypatch):
        reg = UserProviderRegistry([
            _config("OpenAI", api_key_configured=True),
            _config("Anthropic", api_key_configured=False),
        ])
        mock = AsyncMock(return_value=reg)
        monkeypatch.setattr("database.providers.get_all_provider_configurations", mock)
        return reg

    async def test_returns_configured_provider(self, registry):
        result = await get_provider_config(session=make_session(), provider_name="OpenAI", user_id=uuid.uuid4())

        assert result is registry.OpenAI

    @pytest.mark.parametrize("name", ["Anthropic", "Mistral"])
    async def test_raises_400_for_unconfigured_or_unknown(self, registry, name):
        with pytest.raises(HTTPException) as exc:
            await get_provider_config(session=make_session(), provider_name=name, user_id=uuid.uuid4())

        assert exc.value.status_code == 400
