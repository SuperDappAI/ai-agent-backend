from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from langchain.base_language import BaseLanguageModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema.retriever import BaseRetriever
from pydantic import BaseModel

from agent_manager import AgentManager
from document_summarizer import FlexibleDocumentSummarizer
from generative_memory import GenerativeAgentMemory, MemoryType
from memory_summarizer import MemorySummarizer
from rate_limiter import RateLimiter, SyncRateLimiter

rate_limiter = RateLimiter(rate=5, period=1)  # Allow 5 tasks per second
rate_limiter_sync = SyncRateLimiter(rate=5, period=1)

from unittest.mock import AsyncMock

from pydantic import Field


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


class MockFlexibleDocumentSummarizer(FlexibleDocumentSummarizer):
    asummarize = AsyncMock()

    def __init__(self):
        super().__init__(llm=MockLanguageModel())


class MockAgentManager(AgentManager):
    pass


class MockMemorySummarizer(MemorySummarizer):
    save = AsyncMock()
    asummarize = AsyncMock()

    def __init__(self):
        super().__init__(
            rate_limiter=rate_limiter,
            rate_limiter_sync=rate_limiter_sync,
            flexible_document_summarizer=MockFlexibleDocumentSummarizer(),
            agent_manager=MockAgentManager(rate_limiter, rate_limiter_sync),
        )


class MockBaseDocumentCompressor(BaseDocumentCompressor):
    async def acompress_documents(self, *args, **kwargs):
        pass

    def compress_documents(self, *args, **kwargs):
        pass


class MockVectorStore:
    aadd_documents = AsyncMock()


class MockBaseRetriever(BaseRetriever):
    vectorstore: MockVectorStore = MockVectorStore()

    def _get_relevant_documents(self, *args, **kwargs):
        pass


class MockMemoryRetriever(ContextualCompressionRetriever):
    base_compressor: MockBaseDocumentCompressor = Field(
        default_factory=MockBaseDocumentCompressor
    )
    base_retriever: MockBaseRetriever = Field(default_factory=MockBaseRetriever)

    class Config:
        arbitrary_types_allowed = True


@pytest.fixture
def setup_generative_agent_memory():
    llm = MockLanguageModel()
    memory_retriever = MockMemoryRetriever()
    memory_summarizer = MockMemorySummarizer()
    generative_agent_memory = GenerativeAgentMemory(
        rate_limiter=rate_limiter,
        llm=llm,
        memory_retriever=memory_retriever,
        memory_summarizer=memory_summarizer,
        verbose=True,
    )
    return generative_agent_memory


@pytest.mark.asyncio
async def test_add_memory(setup_generative_agent_memory):
    generative_agent_memory = setup_generative_agent_memory
    memory_content = "sample memory"
    conversation_id = "0000000456"
    importance = "high"
    memory_type = MemoryType.CONSCIOUS_MEMORY
    timestamp = datetime.now()

    # Mock the aadd_documents method to return a specific result
    generative_agent_memory.memory_retriever.base_retriever.vectorstore.aadd_documents.return_value = [
        "mock_id"
    ]

    result = await generative_agent_memory.add_memory(
        memory_content, conversation_id, importance, memory_type, now=timestamp
    )

    # Check that the method was called with the correct arguments
    generative_agent_memory.memory_retriever.base_retriever.vectorstore.aadd_documents.assert_called_once()

    # Check that the result is as expected
    assert result == ["mock_id"]
