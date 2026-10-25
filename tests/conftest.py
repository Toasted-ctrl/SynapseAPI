from unittest.mock import patch

import pytest


@pytest.fixture
def mock_config():
    """Mock config with VECTOR_EMBEDDING_MODELS."""
    with patch("providers.vector_embedding.config") as mock:
        mock.VECTOR_EMBEDDING_MODELS = [
            "nomic-embed-text",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
        yield mock


@pytest.fixture
def mock_decrypt():
    """Mock decrypt function."""
    with patch("providers.vector_embedding.decrypt") as mock:
        mock.return_value = "test-api-key"
        yield mock
