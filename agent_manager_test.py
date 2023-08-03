import unittest
from unittest.mock import MagicMock, patch
from agent_manager import AgentManager
from pathlib import Path

class TestAgentManager(unittest.TestCase):
    def setUp(self):
        self.agent_manager = AgentManager()
    
    @patch('agent_manager.GenerativeAgentMemory', autospec=True)
    @patch('agent_manager.OpenAIEmbeddings', autospec=True)
    @patch('agent_manager.TimeWeightedVectorStoreRetriever', autospec=True)
    @patch('agent_manager.QdrantClient', autospec=True)
    def test_create_memory(self, mock_qdrant_client, mock_retriever, mock_embeddings, mock_memory):
        user_id = 'user1'
        path = './storage_memory/user1'
        self.agent_manager.create_memory(user_id, path)
        mock_qdrant_client.assert_called_with(path=path)
        mock_retriever.assert_called()
        mock_memory.assert_called()

    def test_load(self):
        user_id = 'user1'
        with patch.object(self.agent_manager, 'create_memory') as mock_create_memory:
            self.agent_manager.load(user_id)
            mock_create_memory.assert_called_with(user_id, Path(f"{self.agent_manager.dirpath}/{user_id}"))

    @patch('agent_manager.shutil.rmtree', autospec=True)
    def test_delete_memory(self, mock_rmtree):
        user_id = 'user1'
        self.agent_manager.memory[user_id] = MagicMock()
        userpath = Path(f"{self.agent_manager.dirpath}/{user_id}")
        self.agent_manager.delete_memory(user_id)
        mock_rmtree.assert_called_with(userpath)

    def test_push_memory(self):
        user_id = 'user1'
        conversation_id = 'convo1'
        query = 'query'
        llm_response = 'response'
        with patch.object(self.agent_manager, 'load') as mock_load:
            self.agent_manager.push_memory(user_id, conversation_id, query, llm_response)
            mock_load.assert_called_with(user_id)

    def test_pull_memory(self):
        user_id = 'user1'
        convo_id = 'convo1'
        query = 'query'
        with patch.object(self.agent_manager, 'load') as mock_load:
            self.agent_manager.pull_memory(user_id, convo_id, query)
            mock_load.assert_called_with(user_id)

if __name__ == "__main__":
    unittest.main()
