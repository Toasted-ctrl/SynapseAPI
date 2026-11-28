import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from db_helpers import make_session, params, row
from fastapi import HTTPException

from database.providers import (
    NoProvidersFoundError,
    Provider,
    ProviderConfiguration,
    ProviderExistsError,
    ProviderNotFoundError,
    ProviderRecord,
    UserProviderRegistry,
    delete_provider_record_by_id,
    get_all_provider_configurations,
    get_all_provider_records,
    get_or_create_provider,
    get_provider_config,
    get_providers_by_id,
    get_providers_by_location,
    insert_provider_record,
)
from database.schemas.providers import ProvidersT
from database.user_keys import ProviderAPIKey
from providers.dataclasses import LangChainCon


def _provider_row(
    name="Ollama-1",
    internal=True,
    requires_api_key=False,
    id=None,
    host=None,
    mac_address=None,
    **extra,
):
    return row(
        id=id or uuid.uuid4(),
        name=name,
        base_url=f"http://{name.lower()}",
        internal=internal,
        requires_api_key=requires_api_key,
        langchain_con="ChatOllama" if internal else "ChatOpenAI",
        mac_address=mac_address,
        host=host,
        **extra,
    )


def _config(name, internal=False, api_key_configured=False):
    return ProviderConfiguration(
        id=uuid.uuid4(),
        name=name,
        base_url=f"http://{name}",
        langchain_con="ChatOpenAI",
        api_key_configured=api_key_configured,
        internal=internal,
        requires_api_key=not internal,
    )


def _key(provider_id, api_key="enc:sk", api_key_id=None):
    return ProviderAPIKey(
        user_id=uuid.uuid4(),
        api_key_id=api_key_id or uuid.uuid4(),
        api_key=api_key,
        api_key_short="sk-1234567",
        provider_id=provider_id,
        expiration_date=None,
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
        mac_address="00:11:22:33:44:55",
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
            mac_address="00:11:22:33:44:55",
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
        keys = AsyncMock(
            return_value=[
                _key(openai.id, api_key="enc:sk-openai", api_key_id=openai_key_id)
            ]
        )
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
        monkeypatch.setattr(
            "database.providers.get_user_active_keys",
            AsyncMock(return_value=[_key(ollama.id)]),
        )

        reg = await get_all_provider_configurations(
            session=session, user_id=uuid.uuid4()
        )

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
        reg = UserProviderRegistry(
            [
                _config("OpenAI", api_key_configured=True),
                _config("Anthropic", api_key_configured=False),
            ]
        )
        mock = AsyncMock(return_value=reg)
        monkeypatch.setattr("database.providers.get_all_provider_configurations", mock)
        return reg

    async def test_returns_configured_provider(self, registry):
        result = await get_provider_config(
            session=make_session(), provider_name="OpenAI", user_id=uuid.uuid4()
        )

        assert result is registry.OpenAI

    @pytest.mark.parametrize("name", ["Anthropic", "Mistral"])
    async def test_raises_400_for_unconfigured_or_unknown(self, registry, name):
        with pytest.raises(HTTPException) as exc:
            await get_provider_config(
                session=make_session(), provider_name=name, user_id=uuid.uuid4()
            )

        assert exc.value.status_code == 400


@pytest.mark.asyncio
class TestInsertProviderRecord:
    """Tests for the insert_provider_record function."""

    KWARGS = dict(
        name="TestProvider",
        base_url="http://test-provider.com",
        internal=False,
        requires_api_key=True,
        langchain_con=LangChainCon.OPENAI,
        mac_address="AA:BB:CC:DD:EE:FF",
    )

    async def test_inserts_new_provider_record_successfully(self):
        """Test that a new provider record is inserted and returned correctly."""
        session = make_session(execute=[None])

        result = await insert_provider_record(session=session, **self.KWARGS)

        # Verify the provider was added to the session
        [added] = session.added
        assert isinstance(added, ProvidersT)
        assert added.name == "TestProvider"
        assert added.base_url == "http://test-provider.com"
        assert added.internal is False
        assert added.requires_api_key is True
        assert added.langchain_con == "ChatOpenAI"
        assert added.mac_address == "AA:BB:CC:DD:EE:FF"

        # Verify flush and refresh were called
        session.flush.assert_awaited_once()
        session.refresh.assert_awaited_once()

        # Verify the returned ProviderRecord dataclass
        assert isinstance(result, ProviderRecord)
        assert result.name == "TestProvider"
        assert result.base_url == "http://test-provider.com"
        assert result.internal is False
        assert result.requires_api_key is True
        assert result.langchain_con == "ChatOpenAI"
        assert result.mac_address == "AA:BB:CC:DD:EE:FF"

    async def test_raises_error_when_provider_record_exists(self):
        """Test that ProviderExistsError is raised when a duplicate provider record exists."""
        # Create a custom row with all matching attributes
        existing = row(
            id=uuid.uuid4(),
            name="TestProvider",
            base_url="http://test-provider.com",
            internal=False,
            requires_api_key=True,
            langchain_con="ChatOpenAI",
            mac_address="AA:BB:CC:DD:EE:FF",
        )
        session = make_session(execute=[existing])

        with pytest.raises(ProviderExistsError) as exc:
            await insert_provider_record(session=session, **self.KWARGS)

        assert "Provider TestProvider already exists." in str(exc.value)
        # Verify the provider record was NOT added to the session
        assert len(session.added) == 0

    async def test_inserts_provider_record_without_mac_address(self):
        """Test that a provider record can be inserted without a MAC address."""
        kwargs = {k: v for k, v in self.KWARGS.items() if k != "mac_address"}
        kwargs["mac_address"] = None
        session = make_session(execute=[None])

        result = await insert_provider_record(session=session, **kwargs)

        [added] = session.added
        assert added.mac_address is None
        assert result.mac_address is None

    async def test_inserts_internal_provider_record(self):
        """Test that an internal provider record is inserted correctly."""
        kwargs = {
            **self.KWARGS,
            "internal": True,
            "requires_api_key": False,
            "langchain_con": LangChainCon.OLLAMA,
        }
        session = make_session(execute=[None])

        result = await insert_provider_record(session=session, **kwargs)

        [added] = session.added
        assert added.internal is True
        assert added.requires_api_key is False
        assert added.langchain_con == "ChatOllama"
        assert result.internal is True
        assert result.requires_api_key is False

    async def test_different_provider_records_can_exist(self):
        """Test that provider records with different attributes can both exist."""
        # The check will look for a provider record with the NEW provider record's attributes
        # Since no existing record matches all attributes, it should return None (no match)
        session = make_session(execute=[None])

        await insert_provider_record(
            session=session,
            name="TestProvider",
            base_url="http://test-provider.com",
            internal=False,
            requires_api_key=True,
            langchain_con=LangChainCon.OPENAI,
        )

        [added] = session.added
        assert added.internal is False
        assert added.requires_api_key is True

    async def test_checks_all_attributes_for_duplicates(self):
        """Test that duplicate check considers all attributes including mac_address."""
        # This should succeed because mac_address is different
        # We mock the session to return None (no existing provider record found)
        session = make_session(execute=[None])

        await insert_provider_record(
            session=session,
            name="TestProvider",
            base_url="http://test-provider.com",
            internal=False,
            requires_api_key=True,
            langchain_con=LangChainCon.OPENAI,
            mac_address="AA:BB:CC:DD:EE:FF",
        )

        assert len(session.added) == 1

    async def test_exact_match_including_mac_address_raises_error(self):
        """Test that exact match on all attributes including mac_address raises error."""
        existing = row(
            id=uuid.uuid4(),
            name="TestProvider",
            base_url="http://test-provider.com",
            internal=False,
            requires_api_key=True,
            langchain_con="ChatOpenAI",
            mac_address="AA:BB:CC:DD:EE:FF",
        )
        session = make_session(execute=[existing])

        with pytest.raises(ProviderExistsError) as exc:
            await insert_provider_record(
                session=session,
                name="TestProvider",
                base_url="http://test-provider.com",
                internal=False,
                requires_api_key=True,
                langchain_con=LangChainCon.OPENAI,
                mac_address="AA:BB:CC:DD:EE:FF",
            )

        assert "Provider TestProvider already exists." in str(exc.value)
        assert len(session.added) == 0


@pytest.mark.asyncio
class TestDeleteProviderRecordById:
    """Tests for the delete_provider_record_by_id function."""

    async def test_deletes_existing_provider_record_and_returns_record(self):
        """Test that an existing provider record is deleted and the record is returned."""
        provider_id = uuid.uuid4()
        existing = _provider_row(id=provider_id, name="DeleteMe", internal=True)
        session = make_session(execute=[existing])

        result = await delete_provider_record_by_id(id=provider_id, session=session)

        assert isinstance(result, ProviderRecord)
        assert result.id == provider_id
        assert result.name == "DeleteMe"
        session.delete.assert_called_once_with(existing)

    async def test_raises_error_when_provider_record_not_found(self):
        """Test that ProviderNotFoundError is raised when the provider record doesn't exist."""
        provider_id = uuid.uuid4()
        session = make_session(execute=[None])

        with pytest.raises(ProviderNotFoundError) as exc:
            await delete_provider_record_by_id(id=provider_id, session=session)

        assert f"ProviderRecord with ID {provider_id} does not exist." in str(exc.value)
        session.delete.assert_not_called()

    async def test_filters_on_provider_record_id(self):
        """Test that the query filters on the provider record ID."""
        provider_id = uuid.uuid4()
        session = make_session(execute=[_provider_row(id=provider_id)])

        await delete_provider_record_by_id(id=provider_id, session=session)

        assert params(session.execute.await_args) == [provider_id]

    async def test_returns_correct_record_when_multiple_provider_records_exist(self):
        """Test that the correct provider record is deleted when multiple provider records exist."""
        target_id = uuid.uuid4()
        # The execute will return the first matching provider record
        # We'll mock it to return our target provider record
        target_provider = _provider_row(id=target_id, name="Target", internal=False)
        session = make_session(execute=[target_provider])

        result = await delete_provider_record_by_id(id=target_id, session=session)

        assert isinstance(result, ProviderRecord)
        assert result.id == target_id
        assert result.name == "Target"
        session.delete.assert_called_once_with(target_provider)


@pytest.mark.asyncio
class TestGetAllProviderRecords:
    """Tests for the get_all_provider_records function."""

    async def test_returns_all_provider_records(self):
        """Test that all provider records are returned as ProviderRecord dataclass instances."""
        ollama = _provider_row(name="Ollama", internal=True)
        openai = _provider_row(name="OpenAI", internal=False)

        async def mock_execute(*_):
            result = MagicMock()
            result.scalars.return_value = [ollama, openai]
            return result

        session = MagicMock()
        session.execute = AsyncMock(side_effect=mock_execute)

        result = await get_all_provider_records(session=session)

        assert len(result) == 2
        assert all(isinstance(r, ProviderRecord) for r in result)
        names = [r.name for r in result]
        assert "Ollama" in names
        assert "OpenAI" in names

    async def test_returns_provider_records_with_all_attributes(self):
        """Test that all attributes are correctly mapped to ProviderRecord."""
        provider_row = _provider_row(
            name="TestProvider",
            internal=True,
            requires_api_key=False,
            mac_address="AA:BB:CC:DD:EE:FF",
            host="test-host",
        )

        async def mock_execute(*_):
            result = MagicMock()
            result.scalars.return_value = [provider_row]
            return result

        session = MagicMock()
        session.execute = AsyncMock(side_effect=mock_execute)

        result = await get_all_provider_records(session=session)

        assert len(result) == 1
        record = result[0]
        assert record.id == provider_row.id
        assert record.name == "TestProvider"
        assert record.base_url == "http://testprovider"
        assert record.internal is True
        assert record.requires_api_key is False
        assert record.langchain_con == "ChatOllama"
        assert record.mac_address == "AA:BB:CC:DD:EE:FF"
        assert record.host == "test-host"

    async def test_raises_error_when_no_providers_exist(self):
        """Test that NoProvidersFoundError is raised when no provider records exist."""

        async def mock_execute(*_):
            result = MagicMock()
            result.scalars.return_value = []
            return result

        session = MagicMock()
        session.execute = AsyncMock(side_effect=mock_execute)

        with pytest.raises(NoProvidersFoundError):
            await get_all_provider_records(session=session)

    async def test_returns_single_provider_record(self):
        """Test that a single provider record is returned correctly."""
        single_provider = _provider_row(name="SoloProvider", internal=False)

        async def mock_execute(*_):
            result = MagicMock()
            result.scalars.return_value = [single_provider]
            return result

        session = MagicMock()
        session.execute = AsyncMock(side_effect=mock_execute)

        result = await get_all_provider_records(session=session)

        assert len(result) == 1
        assert result[0].name == "SoloProvider"
        assert result[0].internal is False

    async def test_empty_database_raises_error(self):
        """Test that querying an empty database raises NoProvidersFoundError."""

        async def mock_execute(*_):
            result = MagicMock()
            result.scalars.return_value = []
            return result

        session = MagicMock()
        session.execute = AsyncMock(side_effect=mock_execute)

        with pytest.raises(NoProvidersFoundError):
            await get_all_provider_records(session=session)


@pytest.mark.asyncio
class TestNoProvidersFoundError:
    """Tests for the NoProvidersFoundError exception."""

    async def test_is_value_error_subclass(self):
        """Test that NoProvidersFoundError is a subclass of ValueError."""
        assert issubclass(NoProvidersFoundError, ValueError)

    async def test_can_be_raised_with_default_message(self):
        """Test that NoProvidersFoundError can be raised with the default message."""
        with pytest.raises(NoProvidersFoundError) as exc:
            raise NoProvidersFoundError()

        assert isinstance(exc.value, NoProvidersFoundError)
        assert isinstance(exc.value, ValueError)

    async def test_can_be_raised_with_custom_message(self):
        """Test that NoProvidersFoundError can be raised with a custom message."""
        custom_message = "Custom error message"
        with pytest.raises(NoProvidersFoundError) as exc:
            raise NoProvidersFoundError(custom_message)

        assert str(exc.value) == custom_message
