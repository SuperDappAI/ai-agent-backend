import requests

def test_query_plan():
    response = requests.post("http://localhost:8000/query_plan/", data=dict(query="Compare the temperature in Sydney to that of London today"))
    assert response.status_code == 200
    assert "response" in response.json()
    assert "elapsed_time" in response.json()
    print(response.json())

def test_memory_output():
    mock_data = {"user_id": "1", "query": "test", "llm_response": "response", "conversation_id": "1", "importance": "high"}
    response = requests.post("http://localhost:8000/push_memory/", json=mock_data)
    assert response.status_code == 200
    assert "elapsed_time" in response.json()

def test_memory_input():
    mock_data = {"user_id": "1", "query": "test", "conversation_id": "1", "num_semantic_results": 10, "similarity_threshold": 0.72}
    response = requests.post("http://localhost:8000/pull_memory/", json=mock_data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert "elapsed_time" in response.json()

def test_semantic_search_html():
    mock_data = {"action_items": [{"source_url": "http://example.com", "html_doc": "text1"}], "hash": "string", "query": "test", "num_semantic_results": 10, "similarity_threshold": 0.72}
    response = requests.post("http://localhost:8000/semantic_search_html/", json=mock_data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert "elapsed_time" in response.json()

def test_get_functions():
    mock_data = {"action_items": [{"action": "search stocks", "intent": "get price of aapl", "category": "information retrieval"}], "num_semantic_results": 10, "similarity_threshold": 0.72}
    response = requests.post("http://localhost:8000/get_functions/", json=mock_data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert "elapsed_time" in response.json()
    print(response.json())

def test_clear_user_memory():
    mock_data = {"user_id": "1test", "conversation_id": "1test"}
    response = requests.post("http://localhost:8000/clear_conversation/", json=mock_data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert "elapsed_time" in response.json()

def test_test_callback():
    response = requests.get("http://localhost:8000/test_callback/")
    assert response.status_code == 200
    assert "test" in response.json()


