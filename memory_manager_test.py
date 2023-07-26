import unittest
from unittest.mock import MagicMock, patch
from memory_manager import MemoryManager1

class TestMemoryManager1(unittest.TestCase):
    @patch('memory_manager.load_dotenv')
    @patch('memory_manager.os.getenv')
    @patch('memory_manager.threading.Thread')
    @patch('memory_manager.ServiceContext.from_defaults')
    @patch('memory_manager.LLMRerank')
    def test_init(self, mock_llm_rerank, mock_service_context, mock_thread, mock_getenv, mock_load_dotenv):
        mock_llm_rerank.return_value = MagicMock()
        mock_service_context.return_value = MagicMock()

        mm = MemoryManager1()

        mock_load_dotenv.assert_called_once()
        mock_getenv.assert_called_once_with("OPENAI_API_KEY")
        mock_thread.assert_called()
        mm.stop()

    @patch('memory_manager.schedule.run_pending')
    @patch('memory_manager.time.sleep')
    def test_run_continuously(self, mock_sleep, mock_run_pending):
        mock_sleep.side_effect = lambda *args: exit(0)
        memory_manager = MemoryManager1()

        try:
            memory_manager.run_continuously()
        except SystemExit:
            pass
        mock_run_pending.assert_called()
        memory_manager.stop()

    @patch('memory_manager.Path.exists')
    @patch('memory_manager.Path.is_dir')
    @patch('memory_manager.StorageContext.from_defaults')
    @patch('memory_manager.load_index_from_storage')
    def test_load(self, mock_load_index, mock_storage_context, mock_is_dir, mock_exists):
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_load_index.return_value = MagicMock()
        mock_storage_context.return_value = MagicMock()
        user_id = 'user1'

        memory_manager = MemoryManager1()
        memory_manager.load(user_id)

        mock_exists.assert_called()
        mock_is_dir.assert_called()
        mock_load_index.assert_called()
        mock_storage_context.assert_called()
        memory_manager.stop()

    def test_save(self):
        user_id = 'user1'
        memory_manager = MemoryManager1()
        index_mock = MagicMock()
        index_mock.dirty = True  # Here you set the 'dirty' attribute to your MagicMock object
        memory_manager.index[user_id] = index_mock
        memory_manager.save()
        index_mock.storage_context.persist.assert_called()
        memory_manager.stop()

    def test_push_memory(self):
        user_id = 'user1'
        memory_manager = MemoryManager1()
        query = 'Hello, world!'
        llm_response = 'Hello, user!'
        memory_manager.push_memory(user_id, query, llm_response)
        assert memory_manager.index[user_id].dirty is True  # Here you check if 'dirty' is True
        memory_manager.stop()

    def test_pull_memory(self   ):
        user_id = 'user1'
        query = 'Hello, world!'
        memory_manager = MemoryManager1()
        memory_manager.query_engine[user_id] = MagicMock()
        memory_manager.pull_memory(user_id, query)
        memory_manager.query_engine[user_id].query.assert_called()
        memory_manager.stop()

    @patch('memory_manager.shutil.rmtree')
    @patch('memory_manager.Path.exists')
    @patch('memory_manager.Path.is_dir')
    def test_delete_memory(self, mock_is_dir, mock_exists, mock_rmtree):
        user_id = 'user1'
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        memory_manager = MemoryManager1()
        memory_manager.delete_memory(user_id)
        assert user_id not in memory_manager.index
        assert user_id not in memory_manager.query_engine
        mock_rmtree.assert_called()
        memory_manager.stop()

if __name__ == "__main__":
    unittest.main()
