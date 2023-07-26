import unittest
from unittest.mock import patch, MagicMock, call
from functions_manager import FunctionsManager1

class TestFunctionsManager1(unittest.TestCase):
    @patch("functions_manager.schedule")
    @patch("functions_manager.load_dotenv")
    @patch("functions_manager.threading.Thread")
    @patch("functions_manager.ServiceContext")
    @patch("functions_manager.LLMRerank")
    @patch("functions_manager.load_index_from_storage")
    @patch("functions_manager.StorageContext")
    @patch("functions_manager.os.getenv")
    def test_init(self, mock_getenv, mock_storage_context, mock_load_index_from_storage, 
                  mock_llm_rerank, mock_service_context, mock_thread, mock_load_dotenv, mock_schedule):
        mock_load_index_from_storage.return_value = None
        mock_llm_rerank.return_value = MagicMock()
        mock_service_context.from_defaults.return_value = MagicMock()

        fm = FunctionsManager1()

        # Assert that environment is loaded
        mock_load_dotenv.assert_called_once()
        # Assert that getenv is called
        mock_getenv.assert_any_call("OPENAI_API_KEY")
        # Assert that scheduler is set up correctly
        mock_schedule.every.assert_called_once_with(300)
        mock_schedule.every.return_value.to.assert_called_once_with(600)
        mock_schedule.every.return_value.to.return_value.seconds.do.assert_called()
        # Assert that thread is started
        mock_thread.assert_called()
        fm.stop()

    @patch('memory_manager.schedule.run_pending')
    @patch('memory_manager.time.sleep')
    def test_run_continuously(self, mock_sleep, mock_run_pending):
        mock_sleep.side_effect = lambda *args: exit(0)
        functions_manager = FunctionsManager1()

        try:
            functions_manager.run_continuously()
        except SystemExit:
            pass
        #functions_manager.release_locks()  # Assuming the FunctionsManager1 class has a method to release locks
        mock_run_pending.assert_called()


    def test_transform(self):
        fm = FunctionsManager1()
        data = [{"name": "test_name", "description": "test_description"}]
        result = fm.transform(data, "test_category")
        expected_result = [{"name": "test_name", "description": "test_description", "category": "test_category"}]
        self.assertEqual(result, expected_result)
        fm.stop()

    @patch("functions_manager.Document")
    @patch("functions_manager.VectorStoreIndex")
    def test_push_functions(self, mock_index, mock_document):
        
        functions = {
            'informationretrieval_functions': [{'name': 'function1', 'description': 'description1'}],
            'communication_functions': [{'name': 'function2', 'description': 'description2'}],
            'dataprocessing_functions': [{'name': 'function3', 'description': 'description3'}],
            'sensoryperception_functions': [{'name': 'function4', 'description': 'description4'}]
        }
        fm = FunctionsManager1()
        for idx, func_type in enumerate(functions):
            result = fm.push_functions({func_type: functions[func_type]})
            expected_result = [{f"function{idx+1}": 14}]
            self.assertEqual(result, expected_result)
        fm.stop()

    @patch("functions_manager.load_index_from_storage")
    @patch("functions_manager.VectorStoreIndex")
    @patch("functions_manager.LLMRerank")
    @patch("functions_manager.os.path.exists")
    @patch("functions_manager.os.path.isdir")
    def test_load(self, mock_isdir, mock_exists, mock_llm_rerank, mock_index, mock_load_index_from_storage):
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_load_index_from_storage.return_value = MagicMock()
        mock_llm_rerank.return_value = MagicMock()
        mock_index.as_query_engine.return_value = MagicMock()

        fm = FunctionsManager1()
        result = fm.load()
        self.assertFalse(result)
        fm.stop()

    # Similarly, you would write tests for other methods like save, pull_functions and count_tokens
    @patch("functions_manager.tiktoken.encoding_for_model")
    def test_count_tokens(self, mock_encoding_for_model):
        mock_encoding_for_model.return_value = MagicMock()
        mock_encoding_for_model.return_value.encode.side_effect = ['token1', 'token2', 'token3']
        functions = {'informationretrieval_functions': [{'name': 'function1', 'description': 'description1'}]}
        fm = FunctionsManager1()
        result = fm.count_tokens(functions)
        expected_result = [{'function1': 6}]
        self.assertEqual(result, expected_result)
        fm.stop()

    def test_save(self):
        fm = FunctionsManager1()
        doc1 = MagicMock()
        doc2 = MagicMock()
        doc1.dirty = True
        doc2.dirty = True
        fm.index = [doc1, doc2]
        fm.save()
        doc1.storage_context.persist.assert_called_once()
        doc2.storage_context.persist.assert_called_once()
        fm.stop()

    def test_pull_functions(self):
        fm = FunctionsManager1()
        fm.query_engine = MagicMock()
        fm.pull_functions('test_query')
        fm.query_engine.query.assert_called_once_with('test_query')
        fm.stop()

if __name__ == "__main__":
    unittest.main()
