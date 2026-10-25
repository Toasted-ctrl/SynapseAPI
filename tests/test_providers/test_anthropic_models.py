from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.anthropic.models import get_models


class TestGetModels:
    """Tests for the get_models async function."""

    @pytest.fixture
    def mock_decrypt(self):
        """Mock decrypt function to return a test API key."""
        with patch("providers.anthropic.models.decrypt") as mock:
            mock.return_value = "test-anthropic-api-key"
            yield mock

    @pytest.fixture
    def mock_model_config(self):
        """Mock model_config with predefined model lists."""
        with patch("providers.anthropic.models.model_config") as mock:
            mock.chat_completion_models = AsyncMock(
                return_value=[
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                    "claude-2-1",
                ]
            )
            mock.translation_models = AsyncMock(
                return_value=["claude-3-sonnet-20240229", "claude-2-1"]
            )
            mock.vector_embedding_models = AsyncMock(
                return_value=["claude-3-sonnet-20240229"]
            )
            yield mock

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock AsyncAnthropic client."""
        with patch("providers.anthropic.models.AsyncAnthropic") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_successful_model_retrieval(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
    ):
        """Test successful retrieval of models with valid API key."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = [
            MagicMock(id="claude-3-sonnet-20240229"),
            MagicMock(id="claude-3-haiku-20240307"),
            MagicMock(id="claude-2-1"),
            MagicMock(id="claude-instant-1-2"),
        ]
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        result = await get_models("encrypted-api-key-123")

        # Verify decrypt was called with the encrypted key
        mock_decrypt.assert_called_once_with(content="encrypted-api-key-123")

        # Verify client was initialized with decrypted key
        mock_anthropic_client.assert_called_once_with(api_key="test-anthropic-api-key")

        # Verify model filtering
        assert "chat_completion" in result
        assert "translation" in result
        assert "vector_embedding" in result

        # Check chat_completion models
        assert set(result["chat_completion"]) == {
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2-1",
        }

        # Check translation models
        assert set(result["translation"]) == {"claude-3-sonnet-20240229", "claude-2-1"}

        # Check vector_embedding models
        assert result["vector_embedding"] == ["claude-3-sonnet-20240229"]

    @pytest.mark.asyncio
    async def test_authentication_error_returns_empty_lists(
        self, mock_decrypt, mock_anthropic_client
    ):
        """Test that AuthenticationError returns empty lists for all model types."""
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(
            side_effect=Exception("AuthenticationError")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        # Patch AuthenticationError to be raised
        with patch("providers.anthropic.models.AuthenticationError", Exception):
            result = await get_models("invalid-api-key")

        assert result == {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": [],
        }

    @pytest.mark.asyncio
    async def test_empty_model_list(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
    ):
        """Test handling when API returns no models."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        result = await get_models("encrypted-api-key-123")

        assert result == {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": [],
        }

    @pytest.mark.asyncio
    async def test_no_matching_models(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
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
        mock_anthropic_client.return_value = mock_client

        result = await get_models("encrypted-api-key-123")

        assert result == {
            "chat_completion": [],
            "translation": [],
            "vector_embedding": [],
        }

    @pytest.mark.asyncio
    async def test_partial_model_matching(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
    ):
        """Test when only some models match the configured lists."""
        mock_model_config.chat_completion_models = AsyncMock(
            return_value=["claude-3-sonnet-20240229", "model-not-in-api"]
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
            MagicMock(id="claude-3-sonnet-20240229"),
            MagicMock(id="claude-3-haiku-20240307"),
        ]
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        result = await get_models("encrypted-api-key-123")

        assert result["chat_completion"] == ["claude-3-sonnet-20240229"]
        assert result["translation"] == []
        assert result["vector_embedding"] == []

    @pytest.mark.asyncio
    async def test_all_models_match_all_categories(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
    ):
        """Test when a model appears in all three categories."""
        model_name = "claude-3-sonnet-20240229"
        mock_model_config.chat_completion_models = AsyncMock(return_value=[model_name])
        mock_model_config.translation_models = AsyncMock(return_value=[model_name])
        mock_model_config.vector_embedding_models = AsyncMock(return_value=[model_name])

        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = [MagicMock(id=model_name)]
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        result = await get_models("encrypted-api-key-123")

        assert result["chat_completion"] == [model_name]
        assert result["translation"] == [model_name]
        assert result["vector_embedding"] == [model_name]

    @pytest.mark.asyncio
    async def test_decrypt_called_with_correct_parameter(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
    ):
        """Test that decrypt is called with the content parameter."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        test_key = "my-encrypted-key"
        await get_models(test_key)

        mock_decrypt.assert_called_once()
        call_kwargs = mock_decrypt.call_args.kwargs
        assert "content" in call_kwargs
        assert call_kwargs["content"] == test_key

    @pytest.mark.asyncio
    async def test_logging_on_success(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
    ):
        """Test that debug logging occurs on successful fetch."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        with patch("providers.anthropic.models.log") as mock_log:
            await get_models("test-key-123")

            mock_log.debug.assert_called_once()
            call_args = mock_log.debug.call_args[0]
            assert "Fetching Anthropic models" in call_args[0]
            assert "test-key-1" in call_args[0]

    @pytest.mark.asyncio
    async def test_logging_on_authentication_error(
        self, mock_decrypt, mock_anthropic_client
    ):
        """Test that debug logging occurs on AuthenticationError with error message."""
        mock_client = AsyncMock()
        error_message = "Invalid API key provided"
        mock_client.models.list = AsyncMock(side_effect=Exception(error_message))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        with patch("providers.anthropic.models.log") as mock_log:
            with patch("providers.anthropic.models.AuthenticationError", Exception):
                await get_models("invalid-key")

                # Should have both the initial fetch log and the error log
                assert mock_log.debug.call_count == 2
                calls = [str(call) for call in mock_log.debug.call_args_list]
                assert any(
                    "Could not retrieve Anthropic models" in str(call) for call in calls
                )
                assert any(error_message in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_encrypted_key_truncated_in_log(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
    ):
        """Test that only first 10 chars of encrypted key are logged."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        long_key = "a" * 100
        with patch("providers.anthropic.models.log") as mock_log:
            await get_models(long_key)

            call_args = mock_log.debug.call_args[0]
            log_message = call_args[0]
            # Should contain first 10 chars but not the full key
            assert "a" * 10 in log_message
            assert long_key not in log_message

    @pytest.mark.asyncio
    async def test_preserves_model_order(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
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
        mock_anthropic_client.return_value = mock_client

        result = await get_models("encrypted-api-key-123")

        # Order should match the order in the API response, filtered by config
        assert result["chat_completion"] == ["model-c", "model-a", "model-b"]

    @pytest.mark.asyncio
    async def test_returns_correct_dict_structure(
        self, mock_decrypt, mock_model_config, mock_anthropic_client
    ):
        """Test that the return value always has the correct structure."""
        mock_client = AsyncMock()
        mock_models_response = MagicMock()
        mock_models_response.data = []
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_anthropic_client.return_value = mock_client

        result = await get_models("encrypted-api-key-123")

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "chat_completion",
            "translation",
            "vector_embedding",
        }
        assert all(isinstance(v, list) for v in result.values())
