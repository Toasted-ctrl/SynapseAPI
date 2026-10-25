from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from providers.models import get_all_models


class TestGetAllModels:
    """Tests for the get_all_models async function."""

    @pytest.fixture
    def mock_session(self):
        """Mock AsyncSession."""
        return AsyncMock()

    @pytest.fixture
    def mock_user_id(self):
        """Mock user UUID."""
        return UUID("12345678-1234-5678-1234-567812345678")

    @pytest.fixture
    def mock_get_all_provider_configurations(self):
        """Mock get_all_provider_configurations function."""
        with patch("providers.models.get_all_provider_configurations") as mock:
            yield mock

    @pytest.fixture
    def mock_get_models_ollama(self):
        """Mock get_models_ollama function."""
        with patch("providers.models.get_models_ollama") as mock:
            mock.return_value = {
                "chat_completion": ["llama3", "mistral"],
                "translation": [],
                "vector_embedding": [],
            }
            yield mock

    @pytest.fixture
    def mock_get_models_anthropic(self):
        """Mock get_models_anthropic function."""
        with patch("providers.models.get_models_anthropic") as mock:
            mock.return_value = {
                "chat_completion": ["claude-3-sonnet"],
                "translation": ["claude-3-sonnet"],
                "vector_embedding": [],
            }
            yield mock

    @pytest.fixture
    def mock_get_models_google(self):
        """Mock get_models_google function."""
        with patch("providers.models.get_models_google") as mock:
            mock.return_value = {
                "chat_completion": ["gemini-pro"],
                "translation": [],
                "vector_embedding": [],
            }
            yield mock

    @pytest.fixture
    def mock_get_models_openai(self):
        """Mock get_models_openai function."""
        with patch("providers.models.get_models_openai") as mock:
            mock.return_value = {
                "chat_completion": ["gpt-4", "gpt-3.5-turbo"],
                "translation": ["gpt-4"],
                "vector_embedding": ["text-embedding-3-small"],
            }
            yield mock

    @pytest.fixture
    def mock_log(self):
        """Mock log object."""
        with patch("providers.models.log") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_empty_provider_list(
        self, mock_session, mock_user_id, mock_get_all_provider_configurations
    ):
        """Test that empty result is returned when no providers exist."""
        mock_get_all_provider_configurations.return_value = MagicMock(names=[])

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert result == {}

    @pytest.mark.asyncio
    async def test_all_providers_configured(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_ollama,
        mock_get_models_anthropic,
        mock_get_models_google,
        mock_get_models_openai,
    ):
        """Test that all configured providers are included in the result."""
        # Create mock provider configurations
        mock_provider_ollama = MagicMock()
        mock_provider_ollama.name = "Ollama"
        mock_provider_ollama.langchain_con = "ChatOllama"
        mock_provider_ollama.base_url = "http://localhost:11434"
        mock_provider_ollama.requires_api_key = False
        mock_provider_ollama.api_key_configured = True
        mock_provider_ollama.encrypted_api_key = None

        mock_provider_anthropic = MagicMock()
        mock_provider_anthropic.name = "Anthropic"
        mock_provider_anthropic.langchain_con = "ChatAnthropic"
        mock_provider_anthropic.base_url = "https://api.anthropic.com"
        mock_provider_anthropic.requires_api_key = True
        mock_provider_anthropic.api_key_configured = True
        mock_provider_anthropic.encrypted_api_key = "encrypted-anthropic-key"

        mock_provider_google = MagicMock()
        mock_provider_google.name = "Google"
        mock_provider_google.langchain_con = "ChatGoogleGenerativeAI"
        mock_provider_google.base_url = "https://api.google.com"
        mock_provider_google.requires_api_key = True
        mock_provider_google.api_key_configured = True
        mock_provider_google.encrypted_api_key = "encrypted-google-key"

        mock_provider_openai = MagicMock()
        mock_provider_openai.name = "OpenAI"
        mock_provider_openai.langchain_con = "ChatOpenAI"
        mock_provider_openai.base_url = "https://api.openai.com"
        mock_provider_openai.requires_api_key = True
        mock_provider_openai.api_key_configured = True
        mock_provider_openai.encrypted_api_key = "encrypted-openai-key"

        # Mock the registry to iterate over our providers
        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter(
            [
                mock_provider_ollama,
                mock_provider_anthropic,
                mock_provider_google,
                mock_provider_openai,
            ]
        )
        mock_get_all_provider_configurations.return_value = mock_registry

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert "Ollama" in result
        assert "Anthropic" in result
        assert "Google" in result
        assert "OpenAI" in result
        assert result["Ollama"] == {
            "chat_completion": ["llama3", "mistral"],
            "translation": [],
            "vector_embedding": [],
        }
        assert result["Anthropic"] == {
            "chat_completion": ["claude-3-sonnet"],
            "translation": ["claude-3-sonnet"],
            "vector_embedding": [],
        }

    @pytest.mark.asyncio
    async def test_unconfigured_provider_skipped(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_anthropic,
        mock_log,
    ):
        """Test that providers requiring API key but not configured are skipped."""
        mock_provider_anthropic = MagicMock()
        mock_provider_anthropic.name = "Anthropic"
        mock_provider_anthropic.langchain_con = "ChatAnthropic"
        mock_provider_anthropic.base_url = "https://api.anthropic.com"
        mock_provider_anthropic.requires_api_key = True
        mock_provider_anthropic.api_key_configured = False
        mock_provider_anthropic.encrypted_api_key = None

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([mock_provider_anthropic])
        mock_get_all_provider_configurations.return_value = mock_registry

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert result == {}
        mock_log.debug.assert_called_once()
        call_args = mock_log.debug.call_args[0]
        assert "Provider 'Anthropic' not configured" in call_args[0]
        mock_get_models_anthropic.assert_not_called()

    @pytest.mark.asyncio
    async def test_ollama_provider(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_ollama,
    ):
        """Test that ChatOllama provider is handled correctly."""
        mock_provider = MagicMock()
        mock_provider.name = "Ollama"
        mock_provider.langchain_con = "ChatOllama"
        mock_provider.base_url = "http://localhost:11434"
        mock_provider.requires_api_key = False
        mock_provider.api_key_configured = True
        mock_provider.encrypted_api_key = None

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([mock_provider])
        mock_get_all_provider_configurations.return_value = mock_registry

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert "Ollama" in result
        mock_get_models_ollama.assert_called_once_with(
            base_url="http://localhost:11434"
        )

    @pytest.mark.asyncio
    async def test_anthropic_provider(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_anthropic,
    ):
        """Test that ChatAnthropic provider is handled correctly."""
        mock_provider = MagicMock()
        mock_provider.name = "Anthropic"
        mock_provider.langchain_con = "ChatAnthropic"
        mock_provider.base_url = "https://api.anthropic.com"
        mock_provider.requires_api_key = True
        mock_provider.api_key_configured = True
        mock_provider.encrypted_api_key = "encrypted-anthropic-key"

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([mock_provider])
        mock_get_all_provider_configurations.return_value = mock_registry

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert "Anthropic" in result
        mock_get_models_anthropic.assert_called_once_with(
            encrypted_api_key="encrypted-anthropic-key"
        )

    @pytest.mark.asyncio
    async def test_google_provider(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_google,
    ):
        """Test that ChatGoogleGenerativeAI provider is handled correctly."""
        mock_provider = MagicMock()
        mock_provider.name = "Google"
        mock_provider.langchain_con = "ChatGoogleGenerativeAI"
        mock_provider.base_url = "https://api.google.com"
        mock_provider.requires_api_key = True
        mock_provider.api_key_configured = True
        mock_provider.encrypted_api_key = "encrypted-google-key"

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([mock_provider])
        mock_get_all_provider_configurations.return_value = mock_registry

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert "Google" in result
        mock_get_models_google.assert_called_once_with(
            encrypted_api_key="encrypted-google-key"
        )

    @pytest.mark.asyncio
    async def test_openai_provider(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_openai,
    ):
        """Test that ChatOpenAI provider is handled correctly."""
        mock_provider = MagicMock()
        mock_provider.name = "OpenAI"
        mock_provider.langchain_con = "ChatOpenAI"
        mock_provider.base_url = "https://api.openai.com"
        mock_provider.requires_api_key = True
        mock_provider.api_key_configured = True
        mock_provider.encrypted_api_key = "encrypted-openai-key"

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([mock_provider])
        mock_get_all_provider_configurations.return_value = mock_registry

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert "OpenAI" in result
        mock_get_models_openai.assert_called_once_with(
            encrypted_api_key="encrypted-openai-key", base_url="https://api.openai.com"
        )

    @pytest.mark.asyncio
    async def test_unsupported_provider_skipped(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_ollama,
    ):
        """Test that unsupported providers are skipped."""
        mock_provider = MagicMock()
        mock_provider.name = "Mistral"
        mock_provider.langchain_con = "ChatMistral"
        mock_provider.base_url = "https://api.mistral.ai"
        mock_provider.requires_api_key = True
        mock_provider.api_key_configured = True
        mock_provider.encrypted_api_key = "encrypted-mistral-key"

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([mock_provider])
        mock_get_all_provider_configurations.return_value = mock_registry

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert result == {}
        mock_get_models_ollama.assert_not_called()

    @pytest.mark.asyncio
    async def test_connection_error_handled(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_ollama,
        mock_log,
    ):
        """Test that connection errors are handled gracefully."""
        import httpx

        mock_provider = MagicMock()
        mock_provider.name = "Ollama"
        mock_provider.langchain_con = "ChatOllama"
        mock_provider.base_url = "http://localhost:11434"
        mock_provider.requires_api_key = False
        mock_provider.api_key_configured = True
        mock_provider.encrypted_api_key = None

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([mock_provider])
        mock_get_all_provider_configurations.return_value = mock_registry

        # Make get_models_ollama raise a ConnectError
        mock_get_models_ollama.side_effect = httpx.ConnectError("Connection failed")

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert result == {}
        mock_get_models_ollama.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_timeout_handled(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_ollama,
        mock_log,
    ):
        """Test that connection timeouts are handled gracefully."""
        import httpx

        mock_provider = MagicMock()
        mock_provider.name = "Ollama"
        mock_provider.langchain_con = "ChatOllama"
        mock_provider.base_url = "http://localhost:11434"
        mock_provider.requires_api_key = False
        mock_provider.api_key_configured = True
        mock_provider.encrypted_api_key = None

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([mock_provider])
        mock_get_all_provider_configurations.return_value = mock_registry

        # Make get_models_ollama raise a ConnectTimeout
        mock_get_models_ollama.side_effect = httpx.ConnectTimeout(
            "Connection timed out"
        )

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert result == {}
        mock_get_models_ollama.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_providers_mixed_config(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_ollama,
        mock_get_models_anthropic,
    ):
        """Test handling of multiple providers with mixed configurations."""
        # Configured Ollama provider
        mock_provider_ollama = MagicMock()
        mock_provider_ollama.name = "Ollama"
        mock_provider_ollama.langchain_con = "ChatOllama"
        mock_provider_ollama.base_url = "http://localhost:11434"
        mock_provider_ollama.requires_api_key = False
        mock_provider_ollama.api_key_configured = True
        mock_provider_ollama.encrypted_api_key = None

        # Unconfigured Anthropic provider
        mock_provider_anthropic = MagicMock()
        mock_provider_anthropic.name = "Anthropic"
        mock_provider_anthropic.langchain_con = "ChatAnthropic"
        mock_provider_anthropic.base_url = "https://api.anthropic.com"
        mock_provider_anthropic.requires_api_key = True
        mock_provider_anthropic.api_key_configured = False
        mock_provider_anthropic.encrypted_api_key = None

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter(
            [mock_provider_ollama, mock_provider_anthropic]
        )
        mock_get_all_provider_configurations.return_value = mock_registry

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        # Only Ollama should be in the result
        assert "Ollama" in result
        assert "Anthropic" not in result
        mock_get_models_ollama.assert_called_once()
        mock_get_models_anthropic.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_passed_to_get_configurations(
        self, mock_session, mock_user_id, mock_get_all_provider_configurations
    ):
        """Test that session and user_id are passed to get_all_provider_configurations."""
        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([])
        mock_get_all_provider_configurations.return_value = mock_registry

        await get_all_models(session=mock_session, user_id=mock_user_id)

        mock_get_all_provider_configurations.assert_called_once_with(
            session=mock_session, user_id=mock_user_id
        )

    @pytest.mark.asyncio
    async def test_empty_model_lists(
        self,
        mock_session,
        mock_user_id,
        mock_get_all_provider_configurations,
        mock_get_models_ollama,
    ):
        """Test that empty model lists are returned correctly."""
        mock_get_models_ollama.return_value = {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": [],
        }

        mock_provider = MagicMock()
        mock_provider.name = "Ollama"
        mock_provider.langchain_con = "ChatOllama"
        mock_provider.base_url = "http://localhost:11434"
        mock_provider.requires_api_key = False
        mock_provider.api_key_configured = True
        mock_provider.encrypted_api_key = None

        mock_registry = MagicMock()
        mock_registry.__iter__ = lambda self: iter([mock_provider])
        mock_get_all_provider_configurations.return_value = mock_registry

        result = await get_all_models(session=mock_session, user_id=mock_user_id)

        assert result == {
            "Ollama": {"chat_completion": [], "translation": [], "vector_embedding": []}
        }
