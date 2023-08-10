import unittest
import datetime
from generative_memory import GenerativeAgentMemory
from unittest.mock import  patch, AsyncMock
from qdrant_retriever import MemoryType
from unittest import IsolatedAsyncioTestCase
class TestGenerativeAgentMemory(IsolatedAsyncioTestCase):
    @patch('agent_manager.ContextualCompressionRetriever', autospec=True)
    @patch('agent_manager.CohereRerank', autospec=True)
    @patch('agent_manager.Qdrant', autospec=True)
    @patch('agent_manager.QdrantClient', autospec=True)
    @patch('agent_manager.QDrantVectorStoreRetriever', autospec=True)
    @patch('agent_manager.OpenAIEmbeddings', autospec=True)
    @patch('agent_manager.OpenAI', autospec=True)
    def setUp(self, mock_llm, mock_embeddings, mock_retriever, mock_client, mock_qdrant, mock_cohere, mock_compression):
        client = mock_client()
        mock_vectorstore = mock_qdrant(client, "collection", mock_embeddings)
        MockLanguageModel_instance = mock_llm()
    
        compressor = mock_cohere()
        compression_retriever = mock_compression(
            base_compressor=compressor, base_retriever=mock_retriever(
                collection_name="collection", client=client, vectorstore=mock_vectorstore,
            )
        )
        self.memory = GenerativeAgentMemory(
            llm=MockLanguageModel_instance, 
            memory_retriever=compression_retriever
        )
        self.memory.memory_retriever.base_retriever.vectorstore.aadd_documents = AsyncMock(return_value=[])

    async def test_add_memory(self):
        result = await self.memory.add_memory('test memory', 'user', 'conversation', "high", MemoryType.CONSCIOUS_MEMORY, datetime.datetime.now())
        print(f"result {result}")
        self.assertEqual(result, [])

    async def test_add_memories(self):
        result = await self.memory.add_memories(['test memory 1', 'test memory 2'], 'user', 'conversation', ["high", "high"], [MemoryType.CONSCIOUS_MEMORY, MemoryType.CONSCIOUS_MEMORY], datetime.datetime.now())
        self.assertEqual(result, [])

    def test_fetch_memories(self):
        result = self.memory.fetch_memories('test topic', conversation_id="conversation", current_time=datetime.datetime.now())
        self.assertEqual(result, [])

    async def test_pause_to_reflect(self):
        result = await self.memory.pause_to_reflect('test memory', 'user', 'conversation', datetime.datetime.now())
        self.assertEqual(result, [])

    async def test_save_context(self):
        await self.memory.save_context({'query': 'query', 'llm_response': 'llm_response', 'importance': 'high', 'conversation_id': 'conversation', 'user_id': 'user', 'now_key': datetime.datetime.now(), 'payload_conversation_key': 'test conversation'})
        # There is no return value for this method, so we're just checking it runs without errors

    def test_clear(self):
        self.memory.clear("conversation")
        # There is no return value for this method, so we're just checking it runs without errors

    if __name__ == '__main__':
        unittest.main()
