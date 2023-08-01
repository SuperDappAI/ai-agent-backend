import unittest
import datetime
import json
from langchain.schema import Document
from generative_memory import GenerativeAgentMemory
from time_weighted_retriever import TimeWeightedVectorStoreRetriever
from langchain.schema.language_model import BaseLanguageModel

from typing import Any, List, Optional, Type, Iterable, Sequence
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from langchain.schema.messages import BaseMessage
from langchain.schema.prompt import PromptValue
from langchain.callbacks.manager import Callbacks

class MockVectorStore(VectorStore):
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return ["1"] * len(texts)  # return a list of mock IDs, adjust as needed

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        # return a list of mock Documents, adjust as needed
        return [Document(page_content="mock", metadata={}) for _ in range(k)]

    @classmethod
    def from_texts(
        cls: Type['MockVectorStore'],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> 'MockVectorStore':
        return cls()  # return an instance of the mock class, adjust as needed


class MockTimeWeightedVectorStoreRetriever(TimeWeightedVectorStoreRetriever):
    def get_relevant_documents(self, topic):
        return [Document(page_content="relevant document", metadata={"created_at": datetime.datetime.now(), "importance_score": 7})]

    def get_relevant_documents_for_reflection(self, memory_content, conversation):
        return [Document(page_content="relevant document for reflection", metadata={"created_at": datetime.datetime.now(), "importance_score": 9})]

    def add_documents(self, documents):
        return ["document added"]

class MockGeneration:
    def __init__(self, text):
        self.text = text


class LLMResult:
    def __init__(self, generations):
        self.generations = [generations]


class MockBaseMessage(BaseMessage):
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

    def type(self) -> str:
        """Type of the Message, used for serialization."""
        return ""

    @property
    def lc_serializable(self) -> bool:
        """Whether this class is LangChain serializable."""
        return True

class MockLanguageModel(BaseLanguageModel):
    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        return "mocked_prediction"

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        generation = MockGeneration("mocked_prompt")
        return LLMResult([generation])  # pass a list with a single MockGeneration

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        generation = MockGeneration("1;2;3;4")
        return LLMResult([generation])  # pass a list with a single MockGeneration

    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
        return '1;2;3;4'  # mock return

    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> MockBaseMessage:
        return MockBaseMessage("1;2;3;4")

    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> MockBaseMessage:
        return MockBaseMessage("1;2;3;4")
    
    def invoke(self, *args, **kwargs):
        return "mocked_invoke"

class TestGenerativeAgentMemory(unittest.TestCase):
    def setUp(self):
        mock_vectorstore = MockVectorStore()
        MockLanguageModel_instance = MockLanguageModel()
        self.memory = GenerativeAgentMemory(
            llm=MockLanguageModel_instance, 
            memory_retriever=MockTimeWeightedVectorStoreRetriever(vectorstore=mock_vectorstore)
    )

    def test_add_memory(self):
        result = self.memory.add_memory('test memory', 'test conversation', datetime.datetime.now())
        self.assertEqual(result, ['1'])

    def test_add_memories(self):
        result = self.memory.add_memories(['test memory 1', 'test memory 2'], 'test conversation', datetime.datetime.now())
        self.assertEqual(result, ['1', '1'])

    def test_fetch_memories(self):
        result = self.memory.fetch_memories('test topic', current_time=datetime.datetime.now())
        self.assertEqual(result[0].page_content, 'relevant document')

    def test_pause_to_reflect(self):
        result = self.memory.pause_to_reflect('test memory', 'test conversation', datetime.datetime.now())
        self.assertEqual(result, ['1;2;3;4'])

    def test_save_context(self):
        self.memory.save_context({'add_memory_user_key': 'test user', 'add_memory_aida_key': 'test aida', 'now_key': datetime.datetime.now(), 'payload_conversation_key': 'test conversation'})
        # There is no return value for this method, so we're just checking it runs without errors

    def test_clear(self):
        self.memory.clear()
        # There is no return value for this method, so we're just checking it runs without errors

if __name__ == '__main__':
    unittest.main()
