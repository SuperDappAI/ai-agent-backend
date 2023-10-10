import unittest
from unittest.mock import patch, MagicMock
from functions_manager import FunctionsManager, ActionItem, FunctionInput
from rate_limiter import RateLimiter
rate_limiter = RateLimiter(rate=5, period=1) 

class TestFunctionsManager(unittest.TestCase):
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

        fm = FunctionsManager(rate_limiter)

        # Assert that environment is loaded
        mock_load_dotenv.assert_called_once()
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
        functions_manager = FunctionsManager(rate_limiter)

        try:
            functions_manager.run_continuously()
        except SystemExit:
            pass
        #functions_manager.release_locks()  # Assuming the FunctionsManager class has a method to release locks
        mock_run_pending.assert_called()


    def test_transform(self):
        fm = FunctionsManager(rate_limiter)
        user_id = "2"
        data = [{"name": "test_name", "description": "test_description"}]
        result = fm.transform(user_id, data, "test_category")
        expected_result = [{"name": "test_name", "description": "test_description", "category": "test_category"}]
        self.assertEqual(result, expected_result)
        fm.stop()

    @patch("functions_manager.Document")
    @patch("functions_manager.VectorStoreIndex")
    def test_push_functions(self, mock_index, mock_document):
        
        functions = {
            'information_retrieval': [{'name': 'function1', 'description': 'description1'}],
            'communication': [{'name': 'function2', 'description': 'description2'}],
            'data_processing': [{'name': 'function3', 'description': 'description3'}],
            'sensory_perception': [{'name': 'function4', 'description': 'description4'}]
        }
        fm = FunctionsManager(rate_limiter)
        for idx, func_type in enumerate(functions):
            result = fm.push_functions({func_type: functions[func_type]})
            expected_result = [{f"function{idx+1}": 14}]
            self.assertEqual(result[0], expected_result)
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

        fm = FunctionsManager(rate_limiter)
        result = fm.load()
        if result:
            # If load is successful, these functions should be called
            mock_load_index_from_storage.assert_called_once()
        else:
            # If load fails, these functions should not be called
            mock_load_index_from_storage.assert_not_called()

        fm.stop()


    # Similarly, you would write tests for other methods like save, pull_functions and count_tokens
    @patch("functions_manager.tiktoken.encoding_for_model")
    def test_count_tokens(self, mock_encoding_for_model):
        mock_encoding_for_model.return_value = MagicMock()
        mock_encoding_for_model.return_value.encode.side_effect = ['token1', 'token2', 'token3']
        functions = {'information_retrieval': [{'name': 'function1', 'description': 'description1'}]}
        fm = FunctionsManager(rate_limiter)
        result = fm.count_tokens(functions)
        expected_result = [{'function1': 6}]
        self.assertEqual(result, expected_result)
        fm.stop()

    def test_save(self):
        fm = FunctionsManager(rate_limiter)
        fm.index = MagicMock()  # Mock the entire index object
        fm.dirty = True  # Set dirty attribute to True

        fm.save()

        # Assert that persist method was called on the storage_context of the index
        fm.index.storage_context.persist.assert_called_once_with(persist_dir=fm.dirpath)
        self.assertFalse(fm.dirty)  # Ensure dirty flag is set to False after save operation

        fm.stop()

    def test_pull_functions(self):
        fm = FunctionsManager(rate_limiter)
        fm.query_engine = MagicMock()

        # create FunctionInput instance
        action_item = ActionItem(action='test_query', intent='intent_example', category='category_example')
        function_input = FunctionInput(action_items=[action_item])

        fm.pull_functions(function_input)

        # Construct query as per your actual function's implementation
        query = f"action: {action_item.action} intent: {action_item.intent} category: {action_item.category}"

        # Check if query is called with the correct arguments
        fm.query_engine.query.assert_called_once_with(query)

        fm.stop()



if __name__ == "__main__":
    unittest.main()
