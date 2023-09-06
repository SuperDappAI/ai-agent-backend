import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent_manager import AgentManager, MemoryOutput, MemoryInput
from generative_memory import GenerativeAgentMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BaseRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from memory_summarizer import MemorySummarizer
from document_summarizer import FlexibleDocumentSummarizer
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

class MockLanguageModel(BaseLanguageModel):
    async def agenerate_prompt(self, *args, **kwargs):
        pass
    async def apredict(self, *args, **kwargs):
        pass
    async def apredict_messages(self, *args, **kwargs):
        pass
    def generate_prompt(self, *args, **kwargs):
        pass
    def invoke(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        pass
    def predict_messages(self, *args, **kwargs):
        pass

    class Config:
        arbitrary_types_allowed = True

class MockBaseDocumentCompressor(BaseDocumentCompressor):
    async def acompress_documents(self, *args, **kwargs):
        pass
    def compress_documents(self, *args, **kwargs):
        pass

class MockVectorStore:
    aadd_documents = AsyncMock()

class MockBaseRetriever(BaseRetriever):
    vectorstore = MockVectorStore()

    def _get_relevant_documents(self, *args, **kwargs):
        pass

class MockMemoryRetriever(BaseModel):
    base_compressor: MockBaseDocumentCompressor = Field(default_factory=MockBaseDocumentCompressor)
    base_retriever: MockBaseRetriever = Field(default_factory=MockBaseRetriever)

    class Config:
        arbitrary_types_allowed = True

class MockFlexibleDocumentSummarizer(FlexibleDocumentSummarizer):
    pass

class MockMemorySummarizer(MemorySummarizer):
    pass

@pytest.fixture
def setup_agent_manager():
    agent_manager = AgentManager()
    return agent_manager

@pytest.mark.asyncio
async def test_push_memory(setup_agent_manager):
    agent_manager = setup_agent_manager
    memory_output = MemoryOutput(
        api_key="test_api_key",
        user_id="test_user_id",
        query="test_query",
        llm_response="test_llm_response",
        conversation_id="test_conversation_id",
        importance="high"
    )

    with patch.object(agent_manager, 'load', return_value=GenerativeAgentMemory(
        llm=MockLanguageModel(),
        memory_retriever=MockMemoryRetriever(),
        memory_summarizer=MockMemorySummarizer(
            flexible_document_summarizer=MockFlexibleDocumentSummarizer(
                llm=MockLanguageModel()
            ),
            agent_manager=setup_agent_manager
        ),
        verbose=True
    )) as mock_load:
        result = await agent_manager.push_memory(memory_output)

    assert isinstance(result, float)
    mock_load.assert_called_once_with(memory_output.api_key, memory_output.user_id)

def test_create_new_memory_retriever(setup_agent_manager):
    agent_manager = setup_agent_manager

    with patch.object(agent_manager.client, 'create_collection') as mock_create_collection, \
         patch.object(agent_manager.client, 'create_payload_index') as mock_create_payload_index:
        result = agent_manager.create_new_memory_retriever("test_api_key", "test_user_id")

    assert isinstance(result, ContextualCompressionRetriever)
    mock_create_collection.assert_called_once()
    mock_create_payload_index.assert_called_once()