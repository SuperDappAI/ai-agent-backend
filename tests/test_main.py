import pytest
from fastapi.testclient import TestClient
import json
import sys
from os.path import dirname, abspath

# Add the project directory to sys.path
project_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, project_dir)

# Now you can import your app without any issues
from main import app

client = TestClient(app)

from main import app, functions_manager1, web_manager

@pytest.fixture(scope='session', autouse=True)
def shutdown_after_tests():
    yield  # Run tests
    # The code below will run after all tests
    app.on_event("shutdown")
    functions_manager1.stop()
    web_manager.stop()

def test_write_query_plan():
    response = client.post('/query_plan/', data={'query': 'test_query'})
    assert response.status_code == 200
    assert 'response' in response.json()
    assert 'elapsed_time' in response.json()

def test_write_memory_for_user():
    response = client.post('/push_memory/', data={'query': 'test_query', 'llm_response': 'test_response', 'user_id': 'test_user_id', 'conversation_id': 'test_conversation_id'})
    assert response.status_code == 200
    assert 'elapsed_time' in response.json()

def test_delete_html():
    response = client.post('/delete_html/', data={'hash': 'test_hash'})
    assert response.status_code == 200
    assert 'elapsed_time' in response.json()

def test_pull_relevant_memories_for_user():
    response = client.post('/pull_memory/', data={'query': 'test_query', 'user_id': 'test_user_id', 'conversation_id': 'test_conversation_id'})
    assert response.status_code == 200
    assert 'response' in response.json()
    assert 'elapsed_time' in response.json()

# Not Implemented
# def test_pull_latest_memories_for_user():
#     response = client.post('/get_latest_memories/', data={'user_id': 'test_user_id', 'token_count': 10})
#     assert response.status_code == 200
#     assert 'response' in response.json()
#     assert 'elapsed_time' in response.json()

def test_getFunctions():
    client = TestClient(app)

    # Define your payload according to the FunctionInput Pydantic model
    payload = {
        "action_items": [
            {"action": "action_example", "intent": "intent_example", "category": "category_example"}
        ],
        "num_results": 5,
        "similarity_threshold": 0.8
    }

    # Convert the payload to JSON
    payload = json.dumps(payload)

    # Call your endpoint with the test client and the defined payload
    response = client.post('/get_functions/', content=payload)

    # Assert that the response status code is 200 (OK)
    assert response.status_code == 200

    # Assert that the response contains the expected fields
    assert "response" in response.json()
    assert "elapsed_time" in response.json()

# Keep this one off so as not to overwrite the functions.json file

# def test_overwriteFunctions():
#     client = TestClient(app)

#     # Prepare your functionsJson string. This should be a stringified JSON
#     # corresponding to the structure expected by the overwriteFunctions method.
#     functionsJson = json.dumps({
#         "information_retrieval": {
#             # Fill this with the expected structure
#         },
#         # Add more fields if necessary
#     })

#     # Call your endpoint with the test client and the defined payload
#     response = client.post('/overwrite_functions/', data={"functionsJson": functionsJson})

#     # Assert that the response status code is 200 (OK)
#     assert response.status_code == 200

#     # Assert that the response contains the expected fields
#     assert "response" in response.json()
#     assert "elapsed_time" in response.json()

def test_clear_user_memory():
    response = client.post('/clear_conversation/', data={'user_id': 'test_user_id', 'conversation_id': 'test_conversation_id'})
    assert response.status_code == 200
    assert 'response' in response.json()
    assert 'elapsed_time' in response.json()

def test_test_callback():
    response = client.get('/test_callback/')
    assert response.status_code == 200
    assert 'test' in response.json()

