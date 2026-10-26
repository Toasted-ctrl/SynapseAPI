from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.dataclasses import LangChainCon
from providers.vector_embedding import _build_embedding_model, get_embedding


class TestBuildEmbeddingModel:
    """Tests for _build_embedding_model factory function."""

    def test_build_ollama_embedding_without_dimensions(self):
        """Test that Ollama embedding model receives dimensions=None when not provided."""
        with patch("providers.vector_embedding.OllamaEmbeddings") as mock_ollama:
            mock_instance = MagicMock()
            mock_ollama.return_value = mock_instance

            result = _build_embedding_model(
                langchain_con=LangChainCon.OLLAMA,
                model="nomic-embed-text",
                base_url="http://localhost:11434",
            )

            mock_ollama.assert_called_once_with(
                model="nomic-embed-text",
                base_url="http://localhost:11434",
                dimensions=None,
            )
            assert result == mock_instance

    def test_build_ollama_embedding_with_dimensions(self):
        """Test building Ollama embedding model with dimensions."""
        with patch("providers.vector_embedding.OllamaEmbeddings") as mock_ollama:
            mock_instance = MagicMock()
            mock_ollama.return_value = mock_instance

            result = _build_embedding_model(
                langchain_con=LangChainCon.OLLAMA,
                model="nomic-embed-text",
                base_url="http://localhost:11434",
                dimensions=768,
            )

            mock_ollama.assert_called_once_with(
                model="nomic-embed-text",
                base_url="http://localhost:11434",
                dimensions=768,
            )
            assert result == mock_instance

    def test_build_openai_embedding_without_dimensions(self, mock_decrypt):
        """Test that OpenAI embedding model receives dimensions=None when not provided."""
        with patch("providers.vector_embedding.OpenAIEmbeddings") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            result = _build_embedding_model(
                langchain_con=LangChainCon.OPENAI,
                model="text-embedding-3-small",
                base_url="https://api.openai.com/v1",
                encrypted_api_key="encrypted_key_123",
            )

            mock_openai.assert_called_once_with(
                model="text-embedding-3-small",
                base_url="https://api.openai.com/v1",
                dimensions=None,
                openai_api_key="test-api-key",
                check_embedding_ctx_length=True,
            )
            assert result == mock_instance
            mock_decrypt.assert_called_once_with("encrypted_key_123")

    def test_build_openai_embedding_with_dimensions(self, mock_decrypt):
        """Test building OpenAI embedding model with dimensions."""
        with patch("providers.vector_embedding.OpenAIEmbeddings") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            result = _build_embedding_model(
                langchain_con=LangChainCon.OPENAI,
                model="text-embedding-3-large",
                base_url="https://api.openai.com/v1",
                dimensions=3072,
                encrypted_api_key="encrypted_key_123",
            )

            mock_openai.assert_called_once_with(
                model="text-embedding-3-large",
                base_url="https://api.openai.com/v1",
                openai_api_key="test-api-key",
                dimensions=3072,
                check_embedding_ctx_length=True,
            )
            assert result == mock_instance

    def test_openai_embedding_disables_context_check_for_mistral(self, mock_decrypt):
        """Test that Mistral provider disables context length check."""
        with patch("providers.vector_embedding.OpenAIEmbeddings") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            _build_embedding_model(
                langchain_con=LangChainCon.OPENAI,
                model="mistral-embed",
                base_url="https://api.mistral.ai/v1",
                encrypted_api_key="encrypted_key_123",
            )

            call_args = mock_openai.call_args
            assert call_args.kwargs["check_embedding_ctx_length"] is False

    def test_openai_embedding_disables_context_check_for_melious(self, mock_decrypt):
        """Test that Melious provider disables context length check."""
        with patch("providers.vector_embedding.OpenAIEmbeddings") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            _build_embedding_model(
                langchain_con=LangChainCon.OPENAI,
                model="embed-model",
                base_url="https://api.melious.ai/v1",
                encrypted_api_key="encrypted_key_123",
            )

            call_args = mock_openai.call_args
            assert call_args.kwargs["check_embedding_ctx_length"] is False

    def test_openai_embedding_requires_api_key(self, mock_decrypt):
        """Test that OpenAI embedding model raises ValueError without an API key."""
        with patch("providers.vector_embedding.OpenAIEmbeddings") as mock_openai:
            with pytest.raises(ValueError, match="encrypted_api_key is required"):
                _build_embedding_model(
                    langchain_con=LangChainCon.OPENAI,
                    model="text-embedding-3-small",
                    base_url="https://api.openai.com/v1",
                )
            mock_openai.assert_not_called()
            mock_decrypt.assert_not_called()

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Embedding not supported"):
            _build_embedding_model(
                langchain_con=LangChainCon.ANTHROPIC,
                model="text-embedding",
                base_url="https://api.anthropic.com",
            )


class TestGetEmbedding:
    """Tests for get_embedding async function."""

    @pytest.mark.asyncio
    async def test_get_embedding_valid_model(self, mock_embedding_models, mock_decrypt):
        """Test successful embedding retrieval."""
        mock_embedding = AsyncMock()
        mock_embedding.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])

        with patch(
            "providers.vector_embedding._build_embedding_model",
            return_value=mock_embedding,
        ):
            result = await get_embedding(
                langchain_con=LangChainCon.OLLAMA,
                model="nomic-embed-text",
                prompt="test prompt",
                base_url="http://localhost:11434",
            )

            assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_embedding.aembed_query.assert_called_once_with("test prompt")

    @pytest.mark.asyncio
    async def test_get_embedding_with_dimensions(
        self, mock_embedding_models, mock_decrypt
    ):
        """Test embedding retrieval with custom dimensions."""
        mock_embedding = AsyncMock()
        mock_embedding.aembed_query = AsyncMock(return_value=[0.1] * 768)

        with patch(
            "providers.vector_embedding._build_embedding_model",
            return_value=mock_embedding,
        ):
            result = await get_embedding(
                langchain_con=LangChainCon.OPENAI,
                model="text-embedding-3-small",
                prompt="test prompt",
                base_url="https://api.openai.com/v1",
                dimensions=768,
                encrypted_api_key="encrypted_key",
            )

            assert len(result) == 768
            assert all(x == 0.1 for x in result)

    @pytest.mark.asyncio
    async def test_get_embedding_invalid_model(self, mock_embedding_models):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="is not recognized as embedding model"):
            await get_embedding(
                langchain_con=LangChainCon.OLLAMA,
                model="invalid-model",
                prompt="test prompt",
                base_url="http://localhost:11434",
            )

    @pytest.mark.asyncio
    async def test_get_embedding_calls_build_with_correct_params(
        self, mock_embedding_models, mock_decrypt
    ):
        """Test that get_embedding passes correct parameters to _build_embedding_model."""
        mock_embedding = AsyncMock()
        mock_embedding.aembed_query = AsyncMock(return_value=[0.1, 0.2])

        with patch(
            "providers.vector_embedding._build_embedding_model",
            return_value=mock_embedding,
        ) as mock_build:
            await get_embedding(
                langchain_con=LangChainCon.OPENAI,
                model="text-embedding-3-small",
                prompt="test prompt",
                base_url="https://api.openai.com/v1",
                dimensions=1536,
                encrypted_api_key="encrypted_key",
            )

            mock_build.assert_called_once_with(
                langchain_con=LangChainCon.OPENAI,
                model="text-embedding-3-small",
                base_url="https://api.openai.com/v1",
                dimensions=1536,
                encrypted_api_key="encrypted_key",
            )

    @pytest.mark.asyncio
    async def test_get_embedding_returns_empty_list(
        self, mock_embedding_models, mock_decrypt
    ):
        """Test handling of empty embedding response."""
        mock_embedding = AsyncMock()
        mock_embedding.aembed_query = AsyncMock(return_value=[])

        with patch(
            "providers.vector_embedding._build_embedding_model",
            return_value=mock_embedding,
        ):
            result = await get_embedding(
                langchain_con=LangChainCon.OLLAMA,
                model="nomic-embed-text",
                prompt="",
                base_url="http://localhost:11434",
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_get_embedding_large_vector(
        self, mock_embedding_models, mock_decrypt
    ):
        """Test handling of large embedding vectors."""
        large_embedding = [i * 0.001 for i in range(3072)]
        mock_embedding = AsyncMock()
        mock_embedding.aembed_query = AsyncMock(return_value=large_embedding)

        with patch(
            "providers.vector_embedding._build_embedding_model",
            return_value=mock_embedding,
        ):
            result = await get_embedding(
                langchain_con=LangChainCon.OPENAI,
                model="text-embedding-3-large",
                prompt="test prompt",
                base_url="https://api.openai.com/v1",
                dimensions=3072,
                encrypted_api_key="encrypted_key",
            )

            assert len(result) == 3072
            assert result == large_embedding
