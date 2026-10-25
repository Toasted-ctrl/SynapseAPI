from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.openai.models import get_models


class TestGetModels:
    """Tests for the get_models async function."""

    @pytest.fixture
    def mock_decrypt(self):
        """Mock decrypt function to return a test API key."""
        with patch("providers.openai.models.decrypt") as mock:
            mock.return_value = "test-openai-api-key"
            yield mock

    @pytest.fixture
    def mock_model_config(self):
        """Mock model_config with predefined model lists."""
        with patch("providers.openai.models.model_config") as mock:
            mock.chat_completion_models = AsyncMock(
                return_value=[
                    "gpt-4",
                    "gpt-4-0613",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k",
                ]
            )
            mock.translation_models = AsyncMock(return_value=["gpt-4", "gpt-3.5-turbo"])
            mock.vector_embedding_models = AsyncMock(
                return_value=["text-embedding-3-small", "text-embedding-3-large"]
            )
            yield mock

    @pytest.fixture
    def mock_openai_client(self):
        """Mock AsyncOpenAI client."""
        with patch("providers.openai.models.AsyncOpenAI") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_successful_model_retrieval(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test successful retrieval of models with valid API key and base URL."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = [
            MagicMock(id="gpt-4"),
            MagicMock(id="gpt-4-0613"),
            MagicMock(id="gpt-3.5-turbo"),
            MagicMock(id="gpt-3.5-turbo-16k"),
            MagicMock(id="text-embedding-3-small"),
            MagicMock(id="text-embedding-3-large"),
            MagicMock(id="dall-e-3"),
        ]
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        result = await get_models(
            encrypted_api_key="encrypted-api-key-123",
            base_url="https://api.openai.com/v1",
        )

        # Verify decrypt was called with the encrypted key
        mock_decrypt.assert_called_once_with(content="encrypted-api-key-123")

        # Verify client was initialized with decrypted key and base_url
        mock_openai_client.assert_called_once_with(
            api_key="test-openai-api-key", base_url="https://api.openai.com/v1"
        )

        # Verify model filtering
        assert "chat_completion" in result
        assert "translation" in result
        assert "vector_embedding" in result

        # Check chat_completion models
        assert set(result["chat_completion"]) == {
            "gpt-4",
            "gpt-4-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        }

        # Check translation models
        assert set(result["translation"]) == {"gpt-4", "gpt-3.5-turbo"}

        # Check vector_embedding models
        assert set(result["vector_embedding"]) == {
            "text-embedding-3-small",
            "text-embedding-3-large",
        }

    @pytest.mark.asyncio
    async def test_authentication_error_returns_empty_lists(
        self, mock_decrypt, mock_openai_client
    ):
        """Test that AuthenticationError returns empty lists for all model types."""
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(
            side_effect=Exception("AuthenticationError")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        # Patch AuthenticationError to be raised
        with patch("providers.openai.models.AuthenticationError", Exception):
            result = await get_models(
                encrypted_api_key="invalid-api-key",
                base_url="https://api.openai.com/v1",
            )

        assert result == {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": [],
        }

    @pytest.mark.asyncio
    async def test_empty_model_list(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test handling when API returns no models."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        result = await get_models(
            encrypted_api_key="encrypted-api-key-123",
            base_url="https://api.openai.com/v1",
        )

        assert result == {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": [],
        }

    @pytest.mark.asyncio
    async def test_no_matching_models(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test when API returns models but none match the configured model lists."""
        mock_model_config.chat_completion_models = AsyncMock(
            return_value=["model-a", "model-b"]
        )
        mock_model_config.translation_models = AsyncMock(return_value=["model-c"])
        mock_model_config.vector_embedding_models = AsyncMock(return_value=["model-d"])

        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = [
            MagicMock(id="model-x"),
            MagicMock(id="model-y"),
            MagicMock(id="model-z"),
        ]
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        result = await get_models(
            encrypted_api_key="encrypted-api-key-123",
            base_url="https://api.openai.com/v1",
        )

        assert result == {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": [],
        }

    @pytest.mark.asyncio
    async def test_partial_model_matching(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test when only some models match the configured lists."""
        mock_model_config.chat_completion_models = AsyncMock(
            return_value=["gpt-4", "model-not-in-api"]
        )
        mock_model_config.translation_models = AsyncMock(
            return_value=["model-not-in-api"]
        )
        mock_model_config.vector_embedding_models = AsyncMock(
            return_value=["model-not-in-api"]
        )

        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = [
            MagicMock(id="gpt-4"),
            MagicMock(id="gpt-3.5-turbo"),
        ]
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        result = await get_models(
            encrypted_api_key="encrypted-api-key-123",
            base_url="https://api.openai.com/v1",
        )

        assert result["chat_completion"] == ["gpt-4"]
        assert result["translation"] == []
        assert result["vector_embedding"] == []

    @pytest.mark.asyncio
    async def test_all_models_match_all_categories(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test when a model appears in all three categories."""
        model_name = "gpt-4"
        mock_model_config.chat_completion_models = AsyncMock(return_value=[model_name])
        mock_model_config.translation_models = AsyncMock(return_value=[model_name])
        mock_model_config.vector_embedding_models = AsyncMock(return_value=[model_name])

        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = [MagicMock(id=model_name)]
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        result = await get_models(
            encrypted_api_key="encrypted-api-key-123",
            base_url="https://api.openai.com/v1",
        )

        assert result["chat_completion"] == [model_name]
        assert result["translation"] == [model_name]
        assert result["vector_embedding"] == [model_name]

    @pytest.mark.asyncio
    async def test_decrypt_called_with_correct_parameter(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test that decrypt is called with the content parameter."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        test_key = "my-encrypted-key"
        await get_models(
            encrypted_api_key=test_key, base_url="https://api.openai.com/v1"
        )

        mock_decrypt.assert_called_once()
        call_kwargs = mock_decrypt.call_args.kwargs
        assert "content" in call_kwargs
        assert call_kwargs["content"] == test_key

    @pytest.mark.asyncio
    async def test_logging_on_success(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test that debug logging occurs on successful fetch."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        with patch("providers.openai.models.log") as mock_log:
            await get_models(
                encrypted_api_key="test-key-123", base_url="https://api.melious.ai/v1"
            )

            mock_log.debug.assert_called_once()
            call_args = mock_log.debug.call_args[0]
            assert "Fetching models for Provider URL" in call_args[0]
            assert "https://api.melious.ai/v1" in call_args[0]
            assert "test-key-1" in call_args[0]

    @pytest.mark.asyncio
    async def test_logging_on_authentication_error(
        self, mock_decrypt, mock_openai_client
    ):
        """Test that debug logging occurs on AuthenticationError with error message."""
        mock_client = AsyncMock()
        error_message = "Invalid API key provided"
        mock_client.models.list = AsyncMock(side_effect=Exception(error_message))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        with patch("providers.openai.models.log") as mock_log:
            with patch("providers.openai.models.AuthenticationError", Exception):
                await get_models(
                    encrypted_api_key="invalid-key",
                    base_url="https://api.openai.com/v1",
                )

                # Should have both the initial fetch log and the error log
                assert mock_log.debug.call_count == 2
                calls = [str(call) for call in mock_log.debug.call_args_list]
                assert any(
                    "Could not retrieve OpenAI models" in str(call) for call in calls
                )
                assert any(error_message in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_encrypted_key_truncated_in_log(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test that only first 10 chars of encrypted key are logged."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        long_key = "a" * 100
        with patch("providers.openai.models.log") as mock_log:
            await get_models(
                encrypted_api_key=long_key, base_url="https://api.openai.com/v1"
            )

            call_args = mock_log.debug.call_args[0]
            log_message = call_args[0]
            # Should contain first 10 chars but not the full key
            assert "a" * 10 in log_message
            assert long_key not in log_message

    @pytest.mark.asyncio
    async def test_preserves_model_order(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test that model order from API is preserved in results."""
        mock_model_config.chat_completion_models = AsyncMock(
            return_value=["model-a", "model-b", "model-c"]
        )
        mock_model_config.translation_models = AsyncMock(return_value=[])
        mock_model_config.vector_embedding_models = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = [
            MagicMock(id="model-c"),
            MagicMock(id="model-a"),
            MagicMock(id="model-b"),
            MagicMock(id="model-d"),
        ]
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        result = await get_models(
            encrypted_api_key="encrypted-api-key-123",
            base_url="https://api.openai.com/v1",
        )

        # Order should match the order in the API response, filtered by config
        assert result["chat_completion"] == ["model-c", "model-a", "model-b"]

    @pytest.mark.asyncio
    async def test_returns_correct_dict_structure(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test that the return value always has the correct structure."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        result = await get_models(
            encrypted_api_key="encrypted-api-key-123",
            base_url="https://api.openai.com/v1",
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "chat_completion",
            "translation",
            "vector_embedding",
        }
        assert all(isinstance(v, list) for v in result.values())

    @pytest.mark.asyncio
    async def test_base_url_passed_to_client(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test that base_url parameter is correctly passed to AsyncOpenAI client."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        custom_base_url = "https://api.melious.ai/v1"
        await get_models(
            encrypted_api_key="encrypted-api-key-123", base_url=custom_base_url
        )

        mock_openai_client.assert_called_once()
        call_kwargs = mock_openai_client.call_args.kwargs
        assert call_kwargs["base_url"] == custom_base_url

    @pytest.mark.asyncio
    async def test_works_with_different_base_urls(
        self, mock_decrypt, mock_model_config, mock_openai_client
    ):
        """Test that function works with various base URLs (OpenAI, Melious, Z.ai)."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_openai_client.return_value = mock_client

        base_urls = [
            "https://api.openai.com/v1",
            "https://api.melious.ai/v1",
            "https://api.z.ai/v1",
        ]

        for base_url in base_urls:
            mock_openai_client.reset_mock()
            result = await get_models(
                encrypted_api_key="encrypted-api-key-123", base_url=base_url
            )
            assert result == {
                "chat_completion": [],
                "translation": [],
                "vector_embedding": [],
            }
            mock_openai_client.assert_called_once_with(
                api_key="test-openai-api-key", base_url=base_url
            )
