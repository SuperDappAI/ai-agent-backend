import unittest
from unittest.mock import MagicMock, patch
from web_manager import WebManager, HTMLItem, HTMLInput

class TestWebManager(unittest.TestCase):
    @patch('web_manager.load_dotenv')
    @patch('web_manager.os.getenv')
    @patch('web_manager.threading.Thread')
    @patch('web_manager.ServiceContext.from_defaults')
    @patch('web_manager.LLMRerank')
    def test_init(self, mock_llm_rerank, mock_service_context, mock_thread, mock_getenv, mock_load_dotenv):
        mock_llm_rerank.return_value = MagicMock()
        mock_service_context.return_value = MagicMock()

        wm = WebManager()

        mock_load_dotenv.assert_called_once()
        mock_thread.assert_called()
        wm.stop()

    @patch('web_manager.schedule.run_pending')
    @patch('web_manager.time.sleep')
    def test_run_continuously(self, mock_sleep, mock_run_pending):
        mock_sleep.side_effect = lambda *args: exit(0)
        web_manager = WebManager()

        try:
            web_manager.run_continuously()
        except SystemExit:
            pass
        mock_run_pending.assert_called()
        web_manager.stop()

    @patch('web_manager.Path.exists')
    @patch('web_manager.Path.is_dir')
    @patch('web_manager.StorageContext.from_defaults')
    @patch('web_manager.load_index_from_storage')
    def test_load(self, mock_load_index, mock_storage_context, mock_is_dir, mock_exists):
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_load_index.return_value = MagicMock()
        mock_storage_context.return_value = MagicMock()
        hash_key = 'hash1'

        web_manager = WebManager()
        web_manager.load(hash_key)

        mock_exists.assert_called()
        mock_is_dir.assert_called()
        mock_load_index.assert_called()
        mock_storage_context.assert_called()
        web_manager.stop()

    def test_save(self):
        hash_key = 'hash1'
        web_manager = WebManager()
        index_mock = MagicMock()
        index_mock.dirty = True  # Here you set the 'dirty' attribute to your MagicMock object
        web_manager.index[hash_key] = index_mock
        web_manager.save()
        index_mock.storage_context.persist.assert_called()
        web_manager.stop()

    def test_push_html(self):
        hash_key = 'hash1'
        web_manager = WebManager()
        
        html_input = HTMLInput(
            action_items=[
                HTMLItem(source_url='https://example1.com', html_doc='<html><body>Page 1</body></html>'),
                HTMLItem(source_url='https://example2.com', html_doc='<html><body>Page 2</body></html>')
            ],
            hash=hash_key
        )

        web_manager.push_html(html_input)

        assert web_manager.index[hash_key].dirty is True
        web_manager.stop()

    def test_pull_html(self):
        hash_key = 'hash1'
        query = 'Hello, world!'
        web_manager = WebManager()
        web_manager.query_engine[hash_key] = MagicMock()
        web_manager.pull_html(hash_key, query)
        web_manager.query_engine[hash_key].query.assert_called()
        web_manager.stop()

    @patch('web_manager.shutil.rmtree')
    @patch('web_manager.Path.exists')
    @patch('web_manager.Path.is_dir')
    def test_delete_html(self, mock_is_dir, mock_exists, mock_rmtree):
        hash_key = 'hash1'
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        web_manager = WebManager()
        web_manager.delete_html(hash_key)
        assert hash_key not in web_manager.index
        assert hash_key not in web_manager.query_engine
        mock_rmtree.assert_called()
        web_manager.stop()

if __name__ == "__main__":
    unittest.main()
