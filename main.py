import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Form
import json
from agent_manager import AgentManager
from web_manager import WebManager, HTMLInput
from functions_manager import FunctionsManager1, FunctionInput
from queryplan_manager import QueryPlanManager
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://localhost:8000",
    "https://python-api.chatdapp.dev",
]

app = FastAPI()

# Initialize logging
LOGFILE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'app.log')
logging.basicConfig(filename=LOGFILE_PATH, filemode='w',
                    format='%(name)s - %(message)s', force=True, level=logging.INFO)


functions_manager1 = FunctionsManager1()
agent_manager = AgentManager()
web_manager = WebManager()
queryplan_manager = QueryPlanManager()


@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Application shutdown")
    functions_manager1.stop()
    web_manager.stop()

LOGFILE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'app.log')
logging.basicConfig(filename=LOGFILE_PATH, filemode='w',
                    format='%(name)s - %(message)s', force=True)


@app.post('/query_plan/')
async def writeQueryPlan(query: str = Form(...)):
    logging.info(f'Writing query plan for query {query}')
    response, elapsed_time = queryplan_manager.query_plan(query)
    logging.info('Elapsed time for operation: %s',
                 elapsed_time)  # log the elapsed time
    return {'response': response, 'elapsed_time': elapsed_time}


@app.post('/push_memory/')
async def writeMemoryForUser(query: str = Form(...), llm_response: str = Form(...), user_id: str = Form(...), conversation_id: str = Form(...)):
    """Endpoint to push memory for a specific user."""
    logging.info(
        f'Writing memory for user {user_id}, conversation {conversation_id}')
    elapsed_time = agent_manager.push_memory(
        user_id, conversation_id, query, llm_response)
    return {'elapsed_time': elapsed_time}


@app.post('/delete_html/')
async def deleteHTML(hash: str = Form(...)):
    """Endpoint to delete HTML content."""
    logging.info('Deleting HTML')
    elapsed_time = web_manager.delete_html(hash)
    return {'elapsed_time': elapsed_time}


@app.post('/pull_memory/')
async def pullRelevantMemoriesForUser(query: str = Form(...), user_id: str = Form(...), conversation_id: str = Form(...)):
    """Endpoint to pull relevant memories for a specific user."""
    logging.info(
        f'Pulling relevant memories for user {user_id}, conversation {conversation_id}')
    memories, elapsed_time = agent_manager.pull_memory(
        user_id, conversation_id, query)
    return {'response': memories, 'elapsed_time': elapsed_time}


@app.post('/get_latest_memories/')
async def pullLatestMemoriesForUser(user_id: str = Form(...), token_count: int = Form(None)):
    """Endpoint to pull latest memories for a specific user based on token_count."""
    logging.info(f'Pulling latest memories for user {user_id}')
    memories, elapsed_time = agent_manager.get_latest_memories(
        user_id, token_count)
    return {'response': memories, 'elapsed_time': elapsed_time}


@app.post('/semantic_search_html/')
async def semanticSearchHTML(function_input: HTMLInput):
    """Endpoint to conduct a semantic search in HTML content."""
    logging.info('Semantic search HTML')
    results, elapsed_time = await web_manager.search_html(function_input)
    return {'response': results, 'elapsed_time': elapsed_time}


@app.post('/is_html_search_cached/')
async def isHTMLSearchCached(hash_key: str):
    """Endpoint to conduct a semantic search in HTML content."""
    logging.info('Checking if HTML results are cached')
    result, elapsed_time = web_manager.does_hash_exist(hash_key)
    return {'response': result, 'elapsed_time': elapsed_time}


@app.post('/get_functions/')
async def getFunctions(function_input: FunctionInput):
    """Endpoint to get functions based on provided input."""
    logging.info(f'Processing Action Item: {function_input.action_items}')
    result, elapsed_time = functions_manager1.pull_functions(function_input)
    return {'response': result, 'elapsed_time': elapsed_time}


@app.post('/overwrite_functions/')
async def overwriteFunctions(functionsJson: str = Form(...)):
    """Endpoint to overwrite functions."""
    logging.info('Overwriting functions')
    with open('utils/functions.json', 'w') as f:
        f.write(functionsJson)
    with open('utils/functions.json', 'r') as f:
        functionsJson = json.load(f)

    if functionsJson is None or functionsJson['information_retrieval'] is None:
        return {'Reverted': True}

    result, elapsed_time = functions_manager1.push_functions(functionsJson)
    logging.info('Overwrote functions')

    return {'response': result, 'elapsed_time': elapsed_time}


@app.post('/clear_conversation/')
async def clearUserMemory(user_id: str = Form(...), conversation_id: str = Form(...)):
    """Endpoint to clear memory for a specific user/conversation."""
    logging.info(
        f'Clearing user memory for user {user_id} and conversation {conversation_id}')
    response, elapsed_time = agent_manager.clear_conversation(
        user_id, conversation_id)
    return {'response': response, 'elapsed_time': elapsed_time}


@app.get('/test_callback/')
async def test_callback():
    """Test callback endpoint."""
    logging.info('Test callback')
    return {'test': 'test'}
