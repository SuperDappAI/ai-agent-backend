import unittest
from starlette.testclient import TestClient
from main import app
# from pydantic import BaseModel, Field
# from typing import List

# class ActionItem(BaseModel):
#     action: str
#     intent: str
#     category: str

# class FunctionInput(BaseModel):
#     action_items: List[ActionItem] = Field(..., example=[{"action": "action_example", "intent": "intent_example", "category": "category_example"}])
#     num_results: int = Field(..., example=5)
#     similarity_threshold: float = Field(..., example=0.8)

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def tearDown(self):
        pass
    
    def test_writeQueryPlan(self):
        response = self.client.post('/query_plan/', data={'query': 'What is the net worth of Elon Musk compared to that of Jeff Bezos?'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', response.json())
        self.assertIn('elapsed_time', response.json())
        print(response.json())
    
    def test_writeMemoryForUser(self):
        response = self.client.post('/push_memory/', data={
            'message': 'test',
            'llm_response': 'test',
            'user_id': 'test'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('elapsed_time', response.json())
    
    def test_writeMemoryForUser1(self):
        response = self.client.post('/push_memory_1/', data={
            'query': 'test',
            'llm_response': 'test',
            'user_id': 'test',
            'conversation_id': 'test'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('elapsed_time', response.json())
    
    def test_loadHTML(self):
        response = self.client.post('/push_html/', data={
            'html_doc': 'test',
            'source_url': 'test',
            'user_id': 'test'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('elapsed_time', response.json())
    
    def test_deleteHTML(self):
        response = self.client.post('/delete_html/', data={'hash': 'ff0000'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('elapsed_time', response.json())
    
    def test_pullRelevantMemoriesForUser(self):
        response = self.client.post('/pull_memory/', data={
            'query': 'test',
            'user_id': 'test',
            'context': 'test',
            'num_chunks': 2,
            'num_neighbors': 2,
            'similarity_threshold': 0.72
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('memories', response.json())
        self.assertIn('elapsed_time', response.json())
        print(response.json())
    
    def test_pullRelevantMemoriesForUser1(self):
        response = self.client.post('/pull_memory_1/', data={
            'query': 'test',
            'user_id': 'test',
            'conversation_id': 'test'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', response.json())
        self.assertIn('elapsed_time', response.json())
        print(response.json())
    
    def test_pullLatestMemoriesForUser(self):
        response = self.client.post('/get_latest_memories/', data={
            'user_id': 'test',
            'token_count': 100
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', response.json())
        self.assertIn('elapsed_time', response.json())
        print(response.json())
    
    def test_semanticSearchHTML(self):
        response = self.client.post('/semantic_search_html/', data={
            'query': 'test',
            'user_id': 'test',
            'context': 'test',
            'num_results': 2,
            'similarity_threshold': 0.72
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('results', response.json())
        self.assertIn('elapsed_time', response.json())
    
    def test_semanticSearchHTML1(self):
        response = self.client.post('/semantic_search_html_1/', json={
            'html_input': {
                'query': 'test',
                'user_id': 'test',
                'context': 'test',
                'num_results': 2,
                'similarity_threshold': 0.722
            }
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', response.json())
        self.assertIn('elapsed_time', response.json())
     
    def test_getFunctions(self):
        response = self.client.post('/get_functions/', content="application/json",
        data={
                'action_items': [{"action": "action_example", "intent": "intent_example", "category": "category_example"}],
                'num_results': 2,
                'similarity_threshold': 0.72
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', response.json())
        self.assertIn('elapsed_time', response.json())
        print(response.json())
    
    def test_getFunctions1(self):
        response = self.client.post('/get_functions_1/', json={
            'function_input': {
                'action_items': ['test'],
                'num_results': 2,
                'similarity_threshold': 0.72
            }
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', response.json())
        self.assertIn('elapsed_time', response.json())
        print(response.json())
    
    # def test_overwriteFunctions(self):
    #     response = self.client.post('/overwrite_functions/', data={
    #         'functionsJson': 'test',
    #         'examplesJson': 'test'
    #     })
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn('response', response.json())
    
    # def test_overwriteFunctions1(self):
    #     response = self.client.post('/overwrite_functions_1/', data={
    #         'functionsJson': 'test'
    #     })
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn('response', response.json())
    #     self.assertIn('elapsed_time', response.json())
    
    def test_clearUserMemory(self):
        response = self.client.post('/clear_user_memory/', data={'user_id': 'test'})
        self.assertEqual(response.status_code, 200)
    
    def test_clearUserMemory1(self):
        response = self.client.post('/clear_user_memory_1/', data={'user_id': 'test'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('elapsed_time', response.json())
    
    def test_testCallback(self):
        response = self.client.get('/test_callback/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('test', response.json())

if __name__ == '__main__':
    unittest.main()


