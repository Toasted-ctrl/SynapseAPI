import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from database.vector_store import VectorStoreConfig, get_vector_store_settings


def _mock_session(*rows):
    """AsyncSession whose successive execute() calls return the given rows via scalar_one_or_none()."""
    results = []
    for row in rows:
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        results.append(result)
    session = MagicMock()
    session.execute = AsyncMock(side_effect=results)
    return session


@pytest.mark.asyncio
class TestGetVectorStoreSettings:
    """Tests for get_vector_store_settings function."""

    async def test_get_vector_store_settings_success(self):
        """Test successful retrieval of vector store settings."""
        vs_id = uuid.uuid4()

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.vector_store_id = vs_id
        mock_collection.scope = "test_scope"
        mock_collection.e_dimensions = 1536
        mock_collection.e_provider = "openai"
        mock_collection.e_model = "text-embedding-3-small"
        mock_collection.e_api_key = "encrypted_key_123"
        mock_collection.required_filters = ["filter1", "filter2"]

        mock_vs = MagicMock()
        mock_vs.id = vs_id
        mock_vs.vendor = "qdrant"
        mock_vs.base_url = "http://localhost"
        mock_vs.port = 6333

        mock_session = _mock_session(mock_collection, mock_vs)

        result = await get_vector_store_settings("test_scope", mock_session)

        assert isinstance(result, VectorStoreConfig)
        assert result.vs_collection_name == "test_collection"
        assert result.vs_vendor == "qdrant"
        assert result.vs_base_url == "http://localhost"
        assert result.vs_port == 6333
        assert result.e_dimensions == 1536
        assert result.e_provider == "openai"
        assert result.e_model == "text-embedding-3-small"
        assert result.e_api_key == "encrypted_key_123"
        assert result.scope == "test_scope"
        assert result.required_filters == ["filter1", "filter2"]

    async def test_get_vector_store_settings_collection_not_found(self):
        """Test that ValueError is raised when collection is not found."""
        mock_session = _mock_session(None)

        with pytest.raises(
            ValueError, match="Vector Store with name 'nonexistent' does not exist"
        ):
            await get_vector_store_settings("nonexistent", mock_session)

    async def test_get_vector_store_settings_instance_not_found(self):
        """Test that ValueError is raised when vector store instance is not found."""
        vs_id = uuid.uuid4()

        mock_collection = MagicMock()
        mock_collection.vector_store_id = vs_id

        mock_session = _mock_session(mock_collection, None)

        with pytest.raises(
            ValueError, match=f"Vector Store instance with id '{vs_id}' does not exist"
        ):
            await get_vector_store_settings("test_scope", mock_session)

    async def test_get_vector_store_settings_empty_filters(self):
        """Test retrieval with empty required_filters list."""
        vs_id = uuid.uuid4()

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.vector_store_id = vs_id
        mock_collection.scope = "test_scope"
        mock_collection.e_dimensions = 768
        mock_collection.e_provider = "ollama"
        mock_collection.e_model = "nomic-embed-text"
        mock_collection.e_api_key = "ollama_key"
        mock_collection.required_filters = []

        mock_vs = MagicMock()
        mock_vs.vendor = "qdrant"
        mock_vs.base_url = "http://localhost"
        mock_vs.port = 6333

        mock_session = _mock_session(mock_collection, mock_vs)

        result = await get_vector_store_settings("test_scope", mock_session)

        assert result.required_filters == []

    async def test_get_vector_store_settings_with_null_port(self):
        """Test retrieval when port is None."""
        vs_id = uuid.uuid4()

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.vector_store_id = vs_id
        mock_collection.scope = "test_scope"
        mock_collection.e_dimensions = 1536
        mock_collection.e_provider = "openai"
        mock_collection.e_model = "text-embedding-3-large"
        mock_collection.e_api_key = "key"
        mock_collection.required_filters = []

        mock_vs = MagicMock()
        mock_vs.vendor = "qdrant"
        mock_vs.base_url = "http://localhost"
        mock_vs.port = None

        mock_session = _mock_session(mock_collection, mock_vs)

        result = await get_vector_store_settings("test_scope", mock_session)

        assert result.vs_port is None

    async def test_get_vector_store_settings_queries_correct_models(self):
        """Test that the function queries the correct database models."""
        vs_id = uuid.uuid4()

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.vector_store_id = vs_id
        mock_collection.scope = "test_scope"
        mock_collection.e_dimensions = 1536
        mock_collection.e_provider = "openai"
        mock_collection.e_model = "text-embedding-3-small"
        mock_collection.e_api_key = "key"
        mock_collection.required_filters = []

        mock_vs = MagicMock()
        mock_vs.vendor = "qdrant"
        mock_vs.base_url = "http://localhost"
        mock_vs.port = 6333

        mock_session = _mock_session(mock_collection, mock_vs)

        from database.schemas.vector_store import VectorStoreSettingsT
        from database.schemas.vector_store_collections import VectorStoreCollectionT

        await get_vector_store_settings("test_scope", mock_session)

        # Verify query was called with the correct models
        assert mock_session.execute.await_count == 2
        calls = mock_session.execute.await_args_list
        assert (
            calls[0].args[0].column_descriptions[0]["entity"] is VectorStoreCollectionT
        )
        assert calls[1].args[0].column_descriptions[0]["entity"] is VectorStoreSettingsT
