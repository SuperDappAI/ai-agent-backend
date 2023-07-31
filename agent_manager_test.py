import unittest
from unittest.mock import MagicMock, Mock, patch
from your_module_path import Document, AgentManager, GenerativeAgentMemory, TimeWeightedVectorStoreRetriever

class TestAgentManager(unittest.TestCase):
    @patch('agent_manager.load_dotenv')
    @patch('agent_manager.os.getenv')
    def setUp(self, mock_getenv, mock_load_dotenv):
        mock_load_dotenv.return_value = MagicMock()
        mock_getenv.return_value = "api_key"
        self.time_weighted_retriever = MagicMock(spec=TimeWeightedVectorStoreRetriever)
        self.generative_agent_memory = GenerativeAgentMemory(retriever=self.time_weighted_retriever)
        self.agent_manager = AgentManager(memory=self.generative_agent_memory)
        self.agent_manager.index = {}

    def test_retrieve(self):
        self.agent_manager.memory.retriever.get_salient_docs.return_value = [
            Document(id='1', text='Hello, world!', vector=[0.1, 0.2, 0.3], metadata={}),
            Document(id='2', text='Goodbye, world!', vector=[0.4, 0.5, 0.6], metadata={})
        ]
        documents = self.agent_manager.retrieve('user_id', 'query', 'conversation')
        self.assertIsInstance(documents, list)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_add_and_save(self):
        document = Document(id='3', text='New text', vector=[0.7, 0.8, 0.9], metadata={})
        self.agent_manager.add_and_save('user_id', document)
        self.agent_manager.memory.add_and_save.assert_called_once_with(document)

    def test_index_memory(self):
        self.agent_manager.index_memory('user_id', 'conversation', 'reflections', 'salient_doc_ids')
        self.agent_manager.memory.index_memory.assert_called_once()

    def test_search(self):
        mock_query = "test query"
        mock_num_results = 5
        self.agent_manager.memory.retriever.vector_store.search.return_value = [
            Document(id='1', text='Hello, world!', vector=[0.1, 0.2, 0.3], metadata={}),
            Document(id='2', text='Goodbye, world!', vector=[0.4, 0.5, 0.6], metadata={})
        ]
        documents = self.agent_manager.search('user_id', mock_query, mock_num_results)
        self.assertIsInstance(documents, list)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    def test_create_new_memory_retriever(self):
        retriever = self.agent_manager.create_new_memory_retriever("user_id")
        self.agent_manager.memory.create_new_memory_retriever.assert_called_once_with("user_id")

    def test_create_memory(self):
        user_id = 'user1'
        retriever_mock = MagicMock()

        self.agent_manager.create_memory(user_id, retriever_mock)

        self.assertIn(user_id, self.agent_manager.index)
        self.assertEqual(self.agent_manager.index[user_id], retriever_mock)

    def test_load(self):
        user_id = 'user1'
        retriever_mock = MagicMock()

        self.agent_manager.index = {user_id: retriever_mock}
        self.agent_manager.load(user_id)

        retriever_mock.vector_store.load.assert_called()

    def test_push_memory(self):
        user_id = 'user1'
        query = 'Hello, world!'
        llm_response = 'Hello, user!'
        retriever_mock = MagicMock()

        self.agent_manager.index = {user_id: retriever_mock}
        self.agent_manager.push_memory(user_id, query, llm_response)

        retriever_mock.add_and_save.assert_called()

    def test_pull_memory(self):
        user_id = 'user1'
        query = 'Hello, world!'
        retriever_mock = MagicMock()

        self.agent_manager.index = {user_id: retriever_mock}
        self.agent_manager.pull_memory(user_id, query)

        retriever_mock.retrieve.assert_called()

    def test_delete_memory(self):
        user_id = 'user1'
        retriever_mock = MagicMock()

        self.agent_manager.index = {user_id: retriever_mock}
        self.agent_manager.delete_memory(user_id)

        self.assertNotIn(user_id, self.agent_manager.index)
        retriever_mock.vector_store.clear.assert_called()

if __name__ == "__main__":
    unittest.main()
