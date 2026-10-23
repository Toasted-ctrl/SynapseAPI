import pytest
from unittest.mock import MagicMock, patch

from providers.chat_completion import _build_llm
from providers.dataclasses import LangChainCon


class TestBuildLLM:
    """Tests for the _build_llm function."""

    @pytest.fixture
    def mock_decrypt(self):
        """Mock decrypt function to return a test API key."""
        with patch('providers.chat_completion.decrypt') as mock:
            mock.return_value = 'test-api-key'
            yield mock

    @pytest.fixture
    def mock_chat_anthropic(self):
        """Mock ChatAnthropic class."""
        with patch('providers.chat_completion.ChatAnthropic') as mock:
            yield mock

    @pytest.fixture
    def mock_chat_openai(self):
        """Mock ChatOpenAI class."""
        with patch('providers.chat_completion.ChatOpenAI') as mock:
            yield mock

    @pytest.fixture
    def mock_chat_google_generative_ai(self):
        """Mock ChatGoogleGenerativeAI class."""
        with patch('providers.chat_completion.ChatGoogleGenerativeAI') as mock:
            yield mock

    @pytest.fixture
    def mock_chat_ollama(self):
        """Mock ChatOllama class."""
        with patch('providers.chat_completion.ChatOllama') as mock:
            yield mock

    @pytest.fixture
    def mock_log(self):
        """Mock log object."""
        with patch('providers.chat_completion.log') as mock:
            yield mock

    # Tests for ChatAnthropic
    def test_build_anthropic_llm(self, mock_decrypt, mock_chat_anthropic):
        """Test building ChatAnthropic LLM with required API key."""
        mock_instance = MagicMock()
        mock_chat_anthropic.return_value = mock_instance

        result = _build_llm(
            langchain_con="ChatAnthropic",
            model="claude-3-sonnet",
            base_url="https://api.anthropic.com",
            temperature=0.7,
            top_k=5,
            top_p=0.9,
            encrypted_api_key="encrypted-key-123"
        )

        assert result == mock_instance
        mock_chat_anthropic.assert_called_once_with(
            api_key='test-api-key',
            model='claude-3-sonnet',
            base_url='https://api.anthropic.com',
            temperature=0.7,
            top_p=0.9,
            top_k=5
        )
        mock_decrypt.assert_called_once_with('encrypted-key-123')

    def test_build_anthropic_llm_with_enum(self, mock_decrypt, mock_chat_anthropic):
        """Test building ChatAnthropic LLM using LangChainCon enum."""
        mock_instance = MagicMock()
        mock_chat_anthropic.return_value = mock_instance

        result = _build_llm(
            langchain_con=LangChainCon.ANTHROPIC,
            model="claude-3-sonnet",
            base_url="https://api.anthropic.com",
            temperature=0.7,
            top_k=5,
            top_p=0.9,
            encrypted_api_key="encrypted-key-123"
        )

        assert result == mock_instance
        mock_chat_anthropic.assert_called_once()

    def test_build_anthropic_llm_missing_api_key(self, mock_chat_anthropic):
        """Test that ValueError is raised when API key is missing for ChatAnthropic."""
        with pytest.raises(ValueError, match="encrypted_api_key is required for ChatAnthropic"):
            _build_llm(
                langchain_con="ChatAnthropic",
                model="claude-3-sonnet",
                base_url="https://api.anthropic.com",
                temperature=0.7,
                top_k=5,
                top_p=0.9,
                encrypted_api_key=None
            )

    # Tests for ChatOpenAI
    def test_build_openai_llm(self, mock_decrypt, mock_chat_openai):
        """Test building ChatOpenAI LLM with required API key."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        result = _build_llm(
            langchain_con="ChatOpenAI",
            model="gpt-4",
            base_url="https://api.openai.com",
            temperature=0.8,
            top_k=None,
            top_p=0.85,
            encrypted_api_key="encrypted-openai-key"
        )

        assert result == mock_instance
        mock_chat_openai.assert_called_once_with(
            api_key='test-api-key',
            model='gpt-4',
            base_url='https://api.openai.com',
            temperature=0.8,
            top_p=0.85
        )
        # top_k should not be passed to ChatOpenAI
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert 'top_k' not in call_kwargs
        mock_decrypt.assert_called_once_with('encrypted-openai-key')

    def test_build_openai_llm_with_enum(self, mock_decrypt, mock_chat_openai):
        """Test building ChatOpenAI LLM using LangChainCon enum."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        result = _build_llm(
            langchain_con=LangChainCon.OPENAI,
            model="gpt-4",
            base_url="https://api.openai.com",
            temperature=0.8,
            top_k=None,
            top_p=0.85,
            encrypted_api_key="encrypted-openai-key"
        )

        assert result == mock_instance
        mock_chat_openai.assert_called_once()

    def test_build_openai_llm_missing_api_key(self, mock_chat_openai):
        """Test that ValueError is raised when API key is missing for ChatOpenAI."""
        with pytest.raises(ValueError, match="encrypted_api_key is required for ChatOpenAI"):
            _build_llm(
                langchain_con="ChatOpenAI",
                model="gpt-4",
                base_url="https://api.openai.com",
                temperature=0.8,
                top_k=None,
                top_p=0.85,
                encrypted_api_key=None
            )

    # Tests for ChatGoogleGenerativeAI
    def test_build_google_llm(self, mock_decrypt, mock_chat_google_generative_ai):
        """Test building ChatGoogleGenerativeAI LLM with required API key."""
        mock_instance = MagicMock()
        mock_chat_google_generative_ai.return_value = mock_instance

        result = _build_llm(
            langchain_con="ChatGoogleGenerativeAI",
            model="gemini-pro",
            base_url="https://api.google.com",
            temperature=0.9,
            top_k=10,
            top_p=0.95,
            encrypted_api_key="encrypted-google-key"
        )

        assert result == mock_instance
        mock_chat_google_generative_ai.assert_called_once_with(
            api_key='test-api-key',
            model='gemini-pro',
            base_url='https://api.google.com',
            temperature=0.9,
            top_p=0.95,
            top_k=10
        )
        mock_decrypt.assert_called_once_with('encrypted-google-key')

    def test_build_google_llm_missing_api_key(self, mock_chat_google_generative_ai):
        """Test that ValueError is raised when API key is missing for ChatGoogleGenerativeAI."""
        with pytest.raises(ValueError, match="encrypted_api_key is required for ChatGoogleGenerativeAI"):
            _build_llm(
                langchain_con="ChatGoogleGenerativeAI",
                model="gemini-pro",
                base_url="https://api.google.com",
                temperature=0.9,
                top_k=10,
                top_p=0.95,
                encrypted_api_key=None
            )

    # Tests for ChatOllama
    def test_build_ollama_llm(self, mock_chat_ollama):
        """Test building ChatOllama LLM without API key."""
        mock_instance = MagicMock()
        mock_chat_ollama.return_value = mock_instance

        result = _build_llm(
            langchain_con="ChatOllama",
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.7,
            top_k=5,
            top_p=0.9,
            encrypted_api_key=None
        )

        assert result == mock_instance
        mock_chat_ollama.assert_called_once_with(
            model='llama3',
            base_url='http://localhost:11434',
            temperature=0.7,
            top_p=0.9,
            top_k=5
        )
        # decrypt should not be called for Ollama
        with patch('providers.chat_completion.decrypt') as mock_decrypt_check:
            _build_llm(
                langchain_con="ChatOllama",
                model="llama3",
                base_url="http://localhost:11434",
                temperature=0.7,
                top_k=5,
                top_p=0.9,
                encrypted_api_key=None
            )
            mock_decrypt_check.assert_not_called()

    def test_build_ollama_llm_with_enum(self, mock_chat_ollama):
        """Test building ChatOllama LLM using LangChainCon enum."""
        mock_instance = MagicMock()
        mock_chat_ollama.return_value = mock_instance

        result = _build_llm(
            langchain_con=LangChainCon.OLLAMA,
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.7,
            top_k=5,
            top_p=0.9,
            encrypted_api_key=None
        )

        assert result == mock_instance
        mock_chat_ollama.assert_called_once()

    def test_build_ollama_llm_with_api_key(self, mock_chat_ollama):
        """Test that ChatOllama works even if API key is provided (it's ignored)."""
        mock_instance = MagicMock()
        mock_chat_ollama.return_value = mock_instance

        result = _build_llm(
            langchain_con="ChatOllama",
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.7,
            top_k=5,
            top_p=0.9,
            encrypted_api_key="some-key"  # Should be ignored
        )

        assert result == mock_instance
        # Verify encrypted_api_key is not passed to ChatOllama
        call_kwargs = mock_chat_ollama.call_args.kwargs
        assert 'api_key' not in call_kwargs

    # Tests for unsupported provider
    def test_build_unsupported_provider(self, mock_log):
        """Test that ValueError is raised for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider: ChatMistral"):
            _build_llm(
                langchain_con="ChatMistral",
                model="mistral-large",
                base_url="https://api.mistral.ai",
                temperature=0.7,
                top_k=5,
                top_p=0.9,
                encrypted_api_key="some-key"
            )

        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args[0]
        assert "Unsupported LangChain Chat Model: 'ChatMistral'" in call_args[0]

    # Tests for common kwargs
    def test_common_kwargs_passed_to_all_providers(self, mock_decrypt, mock_chat_anthropic, mock_chat_openai, mock_chat_google_generative_ai, mock_chat_ollama):
        """Test that model, base_url, temperature, and top_p are passed to all providers."""
        providers = [
            ("ChatAnthropic", mock_chat_anthropic, "encrypted-key"),
            ("ChatOpenAI", mock_chat_openai, "encrypted-key"),
            ("ChatGoogleGenerativeAI", mock_chat_google_generative_ai, "encrypted-key"),
            ("ChatOllama", mock_chat_ollama, None),
        ]

        for provider_name, mock_class, api_key in providers:
            mock_class.reset_mock()
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            _build_llm(
                langchain_con=provider_name,
                model="test-model",
                base_url="https://test.com",
                temperature=0.5,
                top_k=10 if provider_name != "ChatOpenAI" else None,
                top_p=0.5,
                encrypted_api_key=api_key
            )

            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args.kwargs
            assert call_kwargs['model'] == 'test-model'
            assert call_kwargs['base_url'] == 'https://test.com'
            assert call_kwargs['temperature'] == 0.5
            assert call_kwargs['top_p'] == 0.5

    # Tests for None values
    def test_none_values_handled(self, mock_decrypt, mock_chat_anthropic):
        """Test that None values for temperature, top_k, and top_p are handled."""
        mock_instance = MagicMock()
        mock_chat_anthropic.return_value = mock_instance

        result = _build_llm(
            langchain_con="ChatAnthropic",
            model="claude-3-sonnet",
            base_url="https://api.anthropic.com",
            temperature=None,
            top_k=None,
            top_p=None,
            encrypted_api_key="encrypted-key"
        )

        assert result == mock_instance
        call_kwargs = mock_chat_anthropic.call_args.kwargs
        assert call_kwargs['temperature'] is None
        assert call_kwargs['top_p'] is None
        assert call_kwargs['top_k'] is None

    # Tests for logging
    def test_logging_for_each_provider(self, mock_decrypt, mock_chat_anthropic, mock_chat_openai, mock_chat_google_generative_ai, mock_chat_ollama, mock_log):
        """Test that debug logging occurs for each supported provider."""
        providers = [
            ("ChatAnthropic", mock_chat_anthropic, "encrypted-key"),
            ("ChatOpenAI", mock_chat_openai, "encrypted-key"),
            ("ChatGoogleGenerativeAI", mock_chat_google_generative_ai, "encrypted-key"),
            ("ChatOllama", mock_chat_ollama, None),
        ]

        for provider_name, mock_class, api_key in providers:
            mock_log.reset_mock()
            mock_class.reset_mock()
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            _build_llm(
                langchain_con=provider_name,
                model="test-model",
                base_url="https://test.com",
                temperature=0.5,
                top_k=10,
                top_p=0.5,
                encrypted_api_key=api_key
            )

            mock_log.debug.assert_called_once()
            call_args = mock_log.debug.call_args[0]
            assert f"Constructing Chat Model: {provider_name}" in call_args[0]
