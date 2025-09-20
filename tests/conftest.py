"""Test configuration and fixtures for SuperDappAI tests."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from rate_limiter import RateLimiter, SyncRateLimiter


@pytest.fixture
def rate_limiter():
    """Provide a rate limiter for tests."""
    return RateLimiter(rate=5, period=1)


@pytest.fixture
def rate_limiter_sync():
    """Provide a sync rate limiter for tests."""
    return SyncRateLimiter(rate=5, period=1)


@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key for tests."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        yield "test-api-key"


@pytest.fixture
def mock_cohere_api_key():
    """Mock Cohere API key for tests."""
    with patch.dict(os.environ, {"COHERE_API_KEY": "test-cohere-key"}):
        yield "test-cohere-key"


@pytest.fixture
def mock_qdrant_config():
    """Mock Qdrant configuration for tests."""
    with patch.dict(
        os.environ,
        {"QDRANT_API_KEY": "test-qdrant-key", "QDRANT_URL": "http://localhost:6333"},
    ):
        yield {"api_key": "test-qdrant-key", "url": "http://localhost:6333"}


@pytest.fixture
def mock_mongodb_url():
    """Mock MongoDB URL for tests."""
    with patch.dict(os.environ, {"MONGODB_URL": "mongodb://localhost:27017/test"}):
        yield "mongodb://localhost:27017/test"


@pytest.fixture
def mock_all_env_vars(
    mock_openai_api_key, mock_cohere_api_key, mock_qdrant_config, mock_mongodb_url
):
    """Mock all required environment variables for tests."""
    pass


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for tests."""
    mock_client = Mock()
    mock_client.search = Mock(return_value=[])
    mock_client.delete = Mock()
    mock_client.get_collection = Mock()
    return mock_client


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings for tests."""
    mock = Mock()
    mock.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    mock.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]])
    return mock


@pytest.fixture
def mock_vectorstore():
    """Mock vector store for tests."""
    mock = Mock()
    mock.add_documents = Mock()
    mock.aadd_documents = Mock()
    mock.search = Mock(return_value=[])
    return mock
