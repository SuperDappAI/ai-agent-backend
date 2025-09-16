import pytest
from unittest import mock
from unittest.mock import Mock, patch
from memory_summarizer import MemorySummarizer
from unittest.mock import create_autospec
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from generative_conversation_summarized_memory import GenerativeAgentConversationSummarizedMemory
import os
from rate_limiter import RateLimiter, SyncRateLimiter
rate_limiter = RateLimiter(rate=5, period=1)
rate_limiter_sync = SyncRateLimiter(rate=5, period=1)


@pytest.fixture
def mock_agent_manager():
    mock_agent_manager = Mock()
    mock_agent_manager.client = create_autospec(QdrantClient)
    return mock_agent_manager


@patch('memory_summarizer.Qdrant')
@patch('memory_summarizer.OpenAIEmbeddings')
@patch('memory_summarizer.CohereRerank')
@patch('memory_summarizer.ContextualCompressionRetriever')
@patch('memory_summarizer.QDrantVectorStoreRetriever')
def test_create_new_conversation_summarizer(mock_retriever, mock_compression, mock_rerank, mock_embeddings, mock_qdrant, mock_agent_manager):
    # Arrange
    summarizer = MemorySummarizer(
        rate_limiter, rate_limiter_sync, Mock(), mock_agent_manager)
    api_key = os.getenv("OPENAI_API_KEY")

    # Act
    result = summarizer.create_new_conversation_summarizer(api_key, 'user_id')

    # Assert
    mock_agent_manager.client.create_collection.assert_called_once()
    mock_agent_manager.client.create_payload_index.assert_called_once()
    mock_qdrant.assert_called_once()
    mock_rerank.assert_called_once()
    mock_compression.assert_called_once()
    assert result == mock_compression.return_value


@patch('memory_summarizer.GenerativeAgentConversationSummarizedMemory')
def test_create_summarized_memory(mock_memory, mock_agent_manager):
    # Arrange
    summarizer = MemorySummarizer(
        rate_limiter, rate_limiter_sync, Mock(), mock_agent_manager)
    api_key = os.getenv("OPENAI_API_KEY")

    # Act
    result = summarizer.create_summarized_memory(api_key, 'user_id')

    print(result)
    # # Assert
    mock_memory.assert_called_once_with(
        rate_limiter=rate_limiter,
        llm=mock.ANY,
        memory_retriever=mock.ANY,
        verbose=mock.ANY
    )


@patch('memory_summarizer.GenerativeAgentConversationSummarizedMemory')
def test_load(mock_memory, mock_agent_manager):
    # Arrange
    summarizer = MemorySummarizer(
        rate_limiter, rate_limiter_sync, Mock(), mock_agent_manager)
    api_key = os.getenv("OPENAI_API_KEY")

    # Act
    result = summarizer.load(api_key, 'user_id')

    # Assert
    mock_memory.assert_called_once()
    assert result == mock_memory.return_value
