import pytest
from unittest.mock import MagicMock, patch
from agent_manager import AgentManager, MemoryOutput, MemoryInput
from qdrant_client import QdrantClient
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.retrievers.document_compressors import CohereRerank
from generative_memory import GenerativeAgentMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import Qdrant
from memory_summarizer import MemorySummarizer
from document_summarizer import FlexibleDocumentSummarizer
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from typing import Any, Dict
import os

class TestAgentManager:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.agent_manager = AgentManager()
        yield

    @pytest.mark.asyncio
    async def test_push_memory(self):
        test_api_key = os.getenv("OPENAI_API_KEY")
        memory_output = MemoryOutput(
            api_key=test_api_key,
            user_id="test_user_id",
            query="test_query",
            llm_response="test_llm_response",
            conversation_id="test_conversation_id",
            importance="high"
        )

        with patch.object(self.agent_manager, 'load', return_value=MagicMock()) as mock_load:
            result = await self.agent_manager.push_memory(memory_output)

        assert isinstance(result, float)
        mock_load.assert_called_once_with(memory_output.api_key, memory_output.user_id)

    def test_create_new_memory_retriever(self):

        test_api_key = os.getenv("OPENAI_API_KEY")
        with patch.object(self.agent_manager.client, 'create_collection') as mock_create_collection, \
             patch.object(self.agent_manager.client, 'create_payload_index') as mock_create_payload_index:
            result = self.agent_manager.create_new_memory_retriever(test_api_key, "test_user_id")

        assert isinstance(result, ContextualCompressionRetriever)
        mock_create_collection.assert_called_once()
        mock_create_payload_index.assert_called_once()

    # def test_create_memory(self):
    #     test_api_key = os.getenv("OPENAI_API_KEY")
    #     mock_retriever = MagicMock(base_compressor=MagicMock(), base_retriever=MagicMock())
    #     with patch.object(self.agent_manager, 'create_new_memory_retriever', return_value=mock_retriever) as mock_create_new_memory_retriever:
    #         with patch('generative_memory.GenerativeAgentMemory', return_value=MagicMock(memory_retriever=mock_retriever, llm=MagicMock(), memory_summarizer=MagicMock())) as mock_generative_agent_memory:
    #             result = self.agent_manager.create_memory(test_api_key, "test_user_id")

    #     assert isinstance(result, MagicMock)
    #     mock_create_new_memory_retriever.assert_called_once_with(test_api_key, "test_user_id")

    # @pytest.mark.asyncio
    # async def test_pull_memory(self):
    #     test_api_key = os.getenv("OPENAI_API_KEY")
    #     memory_input = MemoryInput(
    #         api_key=test_api_key,
    #         user_id="test_user_id",
    #         query="test_query",
    #         conversation_id="test_conversation_id",
    #         summary=True
    #     )

    #     async def async_magic_mock(*args, **kwargs):
    #         return {"relevant_summary": "test_summary"}, 0.1

    #     with patch.object(self.agent_manager, 'load_summary', new_callable=AsyncMock, return_value=async_magic_mock()) as mock_load_summary, \
    #          patch.object(self.agent_manager, 'load_memory') as mock_load_memory:
    #         result, time_taken = await self.agent_manager.pull_memory(memory_input)

    #     assert isinstance(result, dict)
    #     assert isinstance(time_taken, float)
    #     mock_load_summary.assert_called_once_with(memory_input)
    #     mock_load_memory.assert_not_called()