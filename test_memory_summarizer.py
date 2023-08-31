import pytest
from unittest import mock
from unittest.mock import AsyncMock
from memory_summarizer import MemorySummarizer
from generative_conversation_summarized_memory import GenerativeAgentConversationSummarizedMemory
from langchain.retrievers import ContextualCompressionRetriever
import os
# from unittest.mock import create_autospec
# from qdrant_client import QdrantClient
# from unittest.mock import MagicMock
# from langchain.schema.language_model import BaseLanguageModel
# from langchain.schema.retriever import BaseRetriever
# from langchain.retrievers.document_compressors.base import (
#     BaseDocumentCompressor,
# )

# class MockDocumentCompressor(BaseDocumentCompressor):
#     async def acompress_documents(self, *args, **kwargs):
#         pass
#     def compress_documents(self, *args, **kwargs):
#         pass

# class MockBaseRetriever(BaseRetriever):
#     async def _get_relevant_documents(self, *args, **kwargs):
#         pass

# class MockRetriever(BaseRetriever):
#     base_compressor = MockDocumentCompressor()
#     base_retriever = MockBaseRetriever()

#     async def _get_relevant_documents(self, *args, **kwargs):
#         pass

# class MockLanguageModel(BaseLanguageModel):
#     async def agenerate_prompt(self, *args, **kwargs):
#         pass
#     async def apredict(self, *args, **kwargs):
#         pass
#     async def apredict_messages(self, *args, **kwargs):
#         pass
#     def generate_prompt(self, *args, **kwargs):
#         pass
#     def invoke(self, *args, **kwargs):
#         pass
#     def predict(self, *args, **kwargs):
#         pass
#     def predict_messages(self, *args, **kwargs):
#         pass

# class MockRetriever(BaseRetriever):
#     base_compressor = MagicMock()
#     base_retriever = MagicMock()

#     async def _get_relevant_documents(self, *args, **kwargs):
#         pass

# @pytest.fixture
# def memory_summarizer():
#     flexible_document_summarizer = mock.Mock()
#     agent_manager = mock.Mock()
#     agent_manager.client = create_autospec(QdrantClient)
#     memory_summarizer = MemorySummarizer(flexible_document_summarizer, agent_manager)
#     memory_summarizer.create_summarized_memory = mock.Mock(return_value=GenerativeAgentConversationSummarizedMemory(
#         llm=MockLanguageModel(),
#         memory_retriever=MockRetriever(),
#         verbose=False
#     ))
#     return memory_summarizer

@pytest.fixture
def memory_summarizer():
    flexible_document_summarizer = mock.Mock()
    agent_manager = mock.Mock()
    memory_summarizer = MemorySummarizer(flexible_document_summarizer, agent_manager)
    memory_summarizer.create_summarized_memory = mock.Mock(return_value=GenerativeAgentConversationSummarizedMemory)
    return memory_summarizer

@pytest.mark.asyncio
async def test_save(memory_summarizer):

    api_key = os.getenv("OPENAI_API_KEY") 
    user_id = 'test_user_id'
    outputs = {'test': 'output'}
    memory = memory_summarizer.load(api_key, user_id)
    memory.save_context = AsyncMock()
    await memory_summarizer.save(api_key, user_id, outputs)
    memory.save_context.assert_called_once_with(outputs)

# def test_create_new_conversation_summarizer(memory_summarizer):

#     api_key = os.getenv("OPENAI_API_KEY") 
#     user_id = 'test_user_id'
#     memory_summarizer.agent_manager.client.create_collection = mock.Mock()
#     memory_summarizer.agent_manager.client.create_payload_index = mock.Mock()
#     summarizer = memory_summarizer.create_new_conversation_summarizer(api_key, user_id)
#     assert isinstance(summarizer, ContextualCompressionRetriever)

# def test_create_summarized_memory(memory_summarizer):

#     api_key = os.getenv("OPENAI_API_KEY") 
#     user_id = 'test_user_id'
#     memory = memory_summarizer.create_summarized_memory(api_key, user_id)
#     assert isinstance(memory, GenerativeAgentConversationSummarizedMemory)