import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Form
import json
from agent_manager import AgentManager, MemoryInput, MemoryOutput
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


functions_manager1 = None
agent_manager = AgentManager()
web_manager = WebManager()
queryplan_manager = QueryPlanManager()


@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutdown")
    agent_manager.stop()
    web_manager.stop()
    if functions_manager1 is not None:
        functions_manager1.stop()
    
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
async def writeMemoryForUser(memory_output: MemoryOutput):
    """Endpoint to push memory for a specific user."""
    logging.info(f'Writing memory for user (importance: {memory_output.importance}) for user {memory_output.user_id}, conversation {memory_output.conversation_id}')
    elapsed_time = await agent_manager.push_memory(memory_output)
    return {'elapsed_time': elapsed_time}


@app.post('/delete_html/')
async def deleteHTML(hash: str = Form(...)):
    """Endpoint to delete HTML content."""
    logging.info('Deleting HTML')
    elapsed_time = web_manager.delete_html(hash)
    return {'elapsed_time': elapsed_time}


@app.post('/pull_memory/')
async def pullRelevantMemoriesForUser(memory_input: MemoryInput):
    """Endpoint to pull relevant memories for a specific user."""
    logging.info(f'Pulling relevant memories for user {memory_input.user_id}, conversation {memory_input.conversation_id}')
    memories, elapsed_time = agent_manager.pull_memory(memory_input)
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
    global functions_manager1  # Declare functions_manager1 as global
    if functions_manager1 is None:
        functions_manager1 = FunctionsManager1()
        await functions_manager1.load()
    logging.info(f'Processing Action Item: {function_input.action_items}')
    result, elapsed_time = await functions_manager1.pull_functions(function_input)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/overwrite_functions/')
async def overwriteFunctions(functionsJson: str = Form(...)):
    """Endpoint to overwrite functions."""
    global functions_manager1  # Declare functions_manager1 as global
    if functions_manager1 is None:
        functions_manager1 = FunctionsManager1()
        await functions_manager1.load()
    logging.info('Overwriting functions')
    with open('utils/functions.json', 'w') as f:
        f.write(functionsJson)
    with open('utils/functions.json', 'r') as f:
        functionsJson = json.load(f)

    if functionsJson is None or functionsJson['information_retrieval'] is None:
        return {'Reverted': True}

    result, elapsed_time = await functions_manager1.push_functions(functionsJson)
    logging.info('Overwrote functions')

    return {'response': result, 'elapsed_time': elapsed_time}


@app.post('/clear_conversation/')
async def clearUserMemory(user_id: str = Form(...), conversation_id: str = Form(...)):
    """Endpoint to clear memory for a specific user/conversation."""
    logging.info(
        f'Clearing user memory for user {user_id} and conversation {conversation_id}')
    response, elapsed_time = agent_manager.clear_conversation(conversation_id)
    return {'response': response, 'elapsed_time': elapsed_time}

@app.get('/test_callback/')
async def test_callback():
    """Test callback endpoint."""
    logging.info('Test callback')
    return {'test': 'test'}
