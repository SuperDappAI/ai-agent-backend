import json
import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import asyncio
import httpx
# from memory import MemoryManager

GPT_MODEL = "gpt-3.5-turbo-0613"
GPT4_MODEL = "gpt-4-0613"

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT4_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(f"function ({message['name']}): {message['content']}\n")
    for formatted_message in formatted_messages:
        print(
            colored(
                formatted_message,
                role_to_color[messages[formatted_messages.index(formatted_message)]["role"]],
            )
        )

# #leaving the async function here to make it easier for building large tests
# async def test_getFunctions(agent_response):
#     if agent_response is None:
#         return
#     if agent_response.json()["choices"][0]["message"]["function_call"] is None:
#         return
#     function_call = agent_response.json()["choices"][0]["message"]["function_call"]
#     print(function_call['arguments'])
#     data = {
#         'categories': ".",
#         'actions': q,
#         'num_results': 2,
#         'similarity_threshold': 0.7 
#     }
#     async with httpx.AsyncClient() as client:  # using httpx AsyncClient
#         response = await client.post("http://127.0.0.1:8000/get_functions/", data=data)  # async post request
#     print(response.status_code)

#     with open('tests/response_agent.json', 'a') as f:
#         f.write('\n')
#         json.dump(response.json(), f)

#     asyncio.run(test_getFunctions())

functions = [
    {
      "name": "getInformationFromMemory",
      "description": "Retrieve historical conversational context through a semantic search. Number of tokens formula: num_chunks x (num_neighbour_chunks+1) x 256 (chunk size). Example: {\"name\": \"getInformationFromMemory\", \"parameters\": {\"conversationID\": \"123456\", \"query\": \"What was discussed earlier?\", \"num_chunks\": 5, \"num_neighbour_chunks\": 2, \"similarity_threshold\": 0.67}}",
      "parameters": {
        "type": "object",
        "properties": {
          "conversationID": {
            "type": "string"
          },
          "query": {
            "type": "string"
          },
          "num_chunks": {
            "description": "How many chunks are considered semantically. More chunks for bigger context. Default is 1.",
            "type": "integer"
          },
          "num_neighbour_chunks": {
            "description": "Add neighbour chunk to every semantically considered chunk. Higher value for general queries or wider range of answers. Default is 1.",
            "type": "integer"
          },
          "similarity_threshold": {
            "description": "Threshold for resulting chunks. Higher threshold for fewer but more precise answers. Default is 0.73.",
            "type": "number"
          }
        },
        "required": ["conversationID", "query", "similarity_threshold"]
      }
    },
    {
      "name": "getInformationFromUser",
      "description": "Ask for more information. Example: {\"name\": \"getInformationFromUser\", \"parameters\": {\"headline\": \"About Your Pets\", \"questions\": [{\"question\": \"What's your pet's name?\", \"input\": {\"placeholder\": \"Pet's name\"}}, {\"question\": \"What type of pet do you have?\", \"radio\": {\"labels\": [\"Dog\", \"Cat\", \"Bird\", \"Other\"]}}, {\"question\": \"When did you get your pet?\", \"date\": {}}, {\"question\": \"Does your pet have any special needs?\", \"checkbox\": {\"labels\": [\"Special Diet\", \"Medication\", \"Accessibility Needs\"]}}]}}",
      "parameters": {
        "type": "object",
        "properties": {
          "headline": {
            "type": "string"
          },
          "questions": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "question": {
                  "type": "string"
                },
                "input": {
                  "type": "object",
                  "properties": {
                    "placeholder": {
                      "type": "string"
                    }
                  }
                },
                "radio": {
                  "type": "object",
                  "properties": {
                    "labels": {
                      "type": "array",
                      "items": {
                        "type": "string"
                      }
                    }
                  }
                },
                "date": {
                  "type": "object"
                },
                "checkbox": {
                  "type": "object",
                  "properties": {
                    "labels": {
                      "type": "array",
                      "items": {
                        "type": "string"
                      }
                    }
                  }
                }
              },
              "required": ["question"]
            }
          }
        },
        "required": ["headline", "questions"]
      }
    },
    {
      "name": "getExternalFunctions",
      "description": "Fetches AiDA's executable functions via action descriptions ('what') and intents ('why'). If no function is returned, no suitable function exists.",
      "parameters": {
        "type": "object",
        "properties": {
          "actions": {
            "type": "string",
            "description": "Comma-separated list of action descriptions that AiDA might perform. The actions should be described as specific tasks or operations that an AI assistant could execute. These should be written in clear and concise language, using function lookup language without using stop words. Example: 'perform map search, execute semantic text analysis'. The order should match with intents and categories."
          },
          "intents": {
            "type": "string",
            "description": "Comma-separated list of intents corresponding to each action. Example: 'find local map, analyse text data'. The order should match with actions and categories."
          },
          "categories": {
            "type": "string",
            "enum": ["Information Retrieval", "Communication", "Data Processing", "Sensory Perception", "Memory"],
            "description": "Comma-separated list of function categories. The order should match with actions and intents."
          }
        },
        "required": ["actions", "categories", "intents"]
      }
    },
    {
      "name": "getUserIDs",
      "description": "Get user IDs based on criteria of given information such as usernames, first names, last names, chat ids, or emails. Each entry in the array can have its own 'global', 'similarity_threshold', and 'num_results' parameters. A specific 'chat_id' can also be provided to filter users from a specific group chat. Example: {'name': 'getUserIDs', 'parameters': {'userquery': [{'userQuery': 'john_doe', 'global': true, 'similarity_threshold': 0.71, 'num_results': 3, 'chat_id': '12345'}, {'userquery': 'jane_doe', 'global': false, 'similarity_threshold': 0.73, 'num_results': 2}]}}",
      "parameters": {
        "type": "object",
        "properties": {
          "userQueries": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "userquery": {
                  "type": "string"
                },
                "global": {
                  "type": "boolean"
                },
                "similarity_threshold": {
                  "type": "number"
                },
                "num_results": {
                  "type": "integer"
                },
                "chat_id": {
                  "type": "string"
                }
              },
              "required": ["userquery"]
            }
          }
        },
        "required": ["userQueries"]
      }
    },
    {
      "name": "summarizeURLs",
      "description": "Creates a summary of the content at the provided URLs. The summary will be concise, highlight the key points, and accept a maximum token size for the summary. URLs must point to text-based content (e.g., TXT, CSV, HTML, JSON, XML, etc.) or PDF files.",
      "parameters": {
        "type": "object",
        "properties": {
          "urls": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "context": {
            "type": "string",
            "description": "Optional. Additional context to guide the summarization."
          },
          "maxTokens": {
            "type": "integer",
            "description": "The maximum token size for the summary."
          }
        },
        "required": ["urls"]
      }
    },
    {
      "name": "searchWebForGeneralInformation",
      "description": "Performs a general web search using the provided query. Can return a direct answer, if available, in the form of an answer box or knowledge graph. Otherwise, returns a set of organic links related to the query. The function can work in conjunction with semanticSearchOnURLs or summarizeURLs for further data processing.",
      "parameters": {
        "type": "object",
        "properties": {
          "q": {
            "type": "string",
            "description": "Defines the query you want to search. You can use anything that you would use in a regular Google search. Supports advanced search query parameters. See the full list of supported advanced search query parameters at https://serpapi.com/advanced-google-query-parameters."
          }
        },
        "required": ["q"]
      }
    },
    {
      "name": "semanticSearchOnURLs",
      "description": "Executes a semantic search on the content from given URLs. The content, which should be text-based or PDF, is indexed into a vector database for semantic matching, with control parameters: num_chunks, num_neighbour_chunks, and similarity_threshold, used similarly to the 'getInformationFromMemory' function.",
      "parameters": {
        "type": "object",
        "properties": {
          "urls": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "num_chunks": {
            "type": "integer"
          },
          "num_neighbour_chunks": {
            "type": "integer"
          },
          "similarity_threshold": {
            "type": "number"
          }
        },
        "required": ["urls"]
      }
    }
  ]

user_input = input("Write your query: ")
messages = []
messages.append({"role": "system", "content": "You're AiDA, an advanced AI assistant designed for non-technical users. Your goal is to perform actions by executing functions, not explaining them. Start with 'getInformationFromMemory' or 'getInformationFromUser' to gather necessary data almost all of the time. After gathering information, use 'getExternalFunctions' to identify and execute suitable functions. You may try 'getExternalFunctions' a few times with varied parameters if responses are not coming back as expected. Follow programming conventions strictly when executing functions. Do not mix dialogue with function calls, and adhere strictly to programming syntax. Be mindful of the token budget to strike a balance between efficiency and quality. You are not allowed to provide dialogue prior to using function calls. Your responses should deliver actionable outcomes, not descriptions of backend actions or functions. When performing semantic searches, a 'similarity_threshold' within 0.65(lenient) to 0.75(strict) is common."})
messages.append({"role": "user", "content": user_input})
chat_response = chat_completion_request(
    messages, functions=functions
)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)


print('------------INITIAL RESPONSE------------')
print(assistant_message)
print(chat_response)
if 'function_call' in assistant_message:
    function_call = chat_response.json()["choices"][0]["message"]["function_call"]
    print(function_call['arguments'])
