from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_embedding_models():
    """Mock the supported vector embedding models."""
    with patch(
        "providers.vector_embedding.model_config.vector_embedding_models",
        new=AsyncMock(
            return_value=[
                "nomic-embed-text",
                "text-embedding-3-small",
                "text-embedding-3-large",
            ]
        ),
    ) as mock:
        yield mock


@pytest.fixture
def mock_decrypt():
    """Mock decrypt function."""
    with patch("providers.vector_embedding.decrypt") as mock:
        mock.return_value = "test-api-key"
        yield mock
