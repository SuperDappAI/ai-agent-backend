import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Form
import json
from agent_manager import AgentManager, MemoryInput, MemoryOutput, ClearMemory
from web_manager import WebManager, HTMLInput, CacheHTML
from doc_manager import DocManager, DocAddInput, DocSearchInput, CacheDoc
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
    "https://api.superdapp.test",
    "https://api.chatdapp.dev",
]

app = FastAPI()

# Initialize logging
LOGFILE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'app.log')
logging.basicConfig(filename=LOGFILE_PATH, filemode='w',
                    format='%(name)s - %(message)s', force=True, level=logging.INFO)


functions_manager = None
agent_manager = AgentManager()
web_manager = WebManager()
queryplan_manager = QueryPlanManager()
doc_manager = DocManager()

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutdown")
    agent_manager.stop()
    web_manager.stop()
    if functions_manager is not None:
        functions_manager.stop()
    
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
async def isHTMLSearchCached(cache_html: CacheHTML):
    """Endpoint to check if HTML content is cached."""
    logging.info('Checking if HTML results are cached')
    result, elapsed_time = web_manager.does_hash_exist(cache_html)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/add_doc/')
async def addDoc(function_input: DocAddInput):
    """Endpoint to conduct add HTML document for doc portal."""
    logging.info('add to Doc Portal')
    results, elapsed_time = await doc_manager.add_doc(function_input)
    return {'response': results, 'elapsed_time': elapsed_time}

@app.post('/is_doc_cached/')
async def isDocCached(cache_html: CacheDoc):
    """Endpoint to check if doc content is cached."""
    logging.info('Checking if doc is cached')
    result, elapsed_time = doc_manager.does_source_exist(cache_html)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/search_doc/')
async def semanticSearchDoc(function_input: DocSearchInput):
    """Endpoint to conduct a semantic search in doc portal."""
    logging.info('Semantic search Doc Portal')
    results, elapsed_time = await doc_manager.search_doc(function_input)
    return {'response': results, 'elapsed_time': elapsed_time}

@app.post('/get_functions/')
async def getFunctions(function_input: FunctionInput):
    """Endpoint to get functions based on provided input."""
    global functions_manager  # Declare functions_manager as global
    if functions_manager is None:
        functions_manager = FunctionsManager1()
        await functions_manager.load()
    logging.info(f'Processing Action Item: {function_input.action_items}')
    result, elapsed_time = await functions_manager.pull_functions(function_input)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/clear_conversation/')
async def clearUserMemory(clear_memory: ClearMemory):
    """Endpoint to clear memory for a specific user/conversation."""
    logging.info(
        f'Clearing user memory for user {clear_memory.user_id} and conversation {clear_memory.conversation_id}')
    response, elapsed_time = agent_manager.clear_conversation(clear_memory)
    return {'response': response, 'elapsed_time': elapsed_time}

@app.get('/test_callback/')
async def test_callback():
    """Test callback endpoint."""
    logging.info('Test callback')
    return {'test': 'test'}
