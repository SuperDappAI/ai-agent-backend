import unittest
import asyncio
from datetime import datetime
from generative_conversation_summarized_memory import GenerativeAgentConversationSummarizedMemory, MemoryType
from agent_manager import AgentManager
from rate_limiter import RateLimiter, SyncRateLimiter
from langchain_community.chat_models import ChatOpenAI
import os


class TestGenerativeAgentConversationSummarizedMemory(unittest.TestCase):
    rate_limiter = RateLimiter(rate=5, period=1)
    rate_limiter_sync = SyncRateLimiter(rate=5, period=1)
    mock_llm = ChatOpenAI()
    mock_retriever = AgentManager(rate_limiter, rate_limiter_sync).create_new_memory_retriever(
        api_key=os.getenv("OPENAI_API_KEY"), user_id="test1")

    @classmethod
    def setUpClass(cls):
        cls.agent_memory = GenerativeAgentConversationSummarizedMemory(
            rate_limiter=cls.rate_limiter,
            llm=cls.mock_llm,
            memory_retriever=cls.mock_retriever,
            verbose=True
        )

    def setUp(self):
        self.loop = asyncio.get_event_loop()

    def test_integration(self):
        user_id = "0000000123"
        conversation_id = "0000000456"
        importance = "high"
        memory_type = MemoryType.CONSCIOUS_MEMORY
        timestamp = datetime.now()

        # Call add_memory method
        memory_content = "sample memory"
        result = self.loop.run_until_complete(self.agent_memory.add_memory(
            memory_content, conversation_id, importance, memory_type, now=timestamp))
        self.assertIsNotNone(result)

        # Call add_memories method
        qa_list = ["question 1", "answer 1"]
        importance_list = ["high", "high"]
        memory_types_list = [MemoryType.CONSCIOUS_MEMORY,
                             MemoryType.CONSCIOUS_MEMORY]
        result = self.loop.run_until_complete(self.agent_memory.add_memories(
            qa_list, conversation_id, importance_list, memory_types_list, now=timestamp))
        self.assertIsNotNone(result)

        # Call save_context method
        result = self.loop.run_until_complete(self.agent_memory.save_context(
            {"query": "sample query", "llm_response": "sample response", "importance": "high", "conversation_id": conversation_id}))
        self.assertIsNotNone(result)

        # Call get_conversation method
        result = self.agent_memory.get_conversation(conversation_id)
        self.assertIsNotNone(result)

        self.agent_memory.clear()


if __name__ == '__main__':
    unittest.main()
