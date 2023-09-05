# import requests
# import json
# import re

# def extract_info_from_log(logfile='tests/response_agent.log'):
#     with open(logfile, 'r') as f:
#         lines = f.readlines()
#     json_str_list = []
#     for line in lines:
#         if line.startswith('root - INFO - {') and 'getExternalFunctions' in line:
#             # Replace single quotes with double quotes
#             correct_json_str = line.replace('root - INFO - ', '').replace("'", "\"")  
#             json_str_list.append(correct_json_str)
#     return [json.loads(i) for i in json_str_list]

# def format_response(response):
#     formatted_response = {
#         "action_items": [{
#             "action": response['actions'], 
#             "intent": response['intents'], 
#             "category": response['categories']
#         }], 
#         "num_semantic_results": 10, 
#         "similarity_threshold": 0.72
#     }
#     return formatted_response

# def get_functions(mock_data=None):
#     log_data_list = extract_info_from_log()
#     if mock_data is None:
#         mock_data = log_data_list
#     for data in mock_data:
#         response = requests.post("http://localhost:8000/get_functions/", json=data)
#         assert response.status_code == 200
#         j_response = response.json()
#         formatted_j_response = format_response(j_response['response'])
#         print(formatted_j_response)

# get_functions()