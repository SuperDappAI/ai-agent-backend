import os
import logging
from typing import List
from dotenv import load_dotenv
import pinecone
from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel, Field
import signal
import sys
import atexit
from memory_search import MemoryManager
from memory_manager import MemoryManager1
from web_manager import WebManager
from custom_text_loader import TextLoader
from functions_endpoint import FunctionsManager
from functions_manager import FunctionsManager1
from queryplan_manager import QueryPlanManager
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

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

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY)

functions_manager1 = FunctionsManager1()
memory_manager1 = MemoryManager1()
web_manager = WebManager()
queryplan_manager = QueryPlanManager()
# register the stop method to be called on exit
atexit.register(functions_manager1.stop)
atexit.register(memory_manager1.stop)
atexit.register(web_manager.stop)

# define a handler for the signals
def signal_handler(signum, frame):
    print(f"Caught signal {signum}, stopping...")
    functions_manager1.stop()
    memory_manager1.stop()
    web_manager.stop()
    
# register the signal handler for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
    
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
    return {'results': response, 'elapsed_time': elapsed_time}

@app.post('/push_memory/')
async def writeMemoryForUser(message: str = Form(...), llm_response: str = Form(...), user_id: str = Form(...)):
    logging.info(f'Writing memory for user {user_id}')
    memory_manager = MemoryManager(user_id)
    elapsed_time = memory_manager.push_memory(message, llm_response)
    logging.info('Pushed memory for user %s, message: %s, response: %s',
                 user_id, message, llm_response)  # log the data push
    logging.info('Elapsed time for operation: %s',
                 elapsed_time)  # log the elapsed time

    return {'elapsed_time': elapsed_time}

@app.post('/push_memory_1/')
async def writeMemoryForUser(query: str = Form(...), llm_response: str = Form(...), user_id: str = Form(...)):
    """Endpoint to push memory for a specific user."""
    logging.info(f'Writing memory for user {user_id}')
    elapsed_time = memory_manager1.push_memory(user_id, query, llm_response)
    logging.info(f'Pushed memory for user {user_id}, query: {query}, response: {llm_response}')  # log the data push
    logging.info(f'Elapsed time for operation: {elapsed_time}')  # log the elapsed time
    return {'elapsed_time': elapsed_time}

@app.post('/push_html/')
async def loadHTML(html_doc: str = Form(...), source_url: str = Form(...), user_id: str = Form(...)):
    logging.info(f'Loading HTML')
    # save file to temporary folder
    with open(f'{user_id}.txt', 'w') as f:
        f.write(html_doc)
        f.close()
    # load file from temporary folder
    loader = TextLoader(f'{user_id}.txt',metadata={'source_url': source_url})
    docs = loader.load()
    # return docs
    logging.info(f'Loaded HTML')
    memory_manager = MemoryManager(user_id=user_id)
    logging.info(f'Pushing HTML to Pinecone')
    elapsed_time = memory_manager.split_and_push_webpage(docs)
    os.remove(f'{user_id}.txt')
    logging.info(f'Pushed HTML to Pinecone')
    logging.info('Elapsed time for operation: %s',
                 elapsed_time)  # log the elapsed time

    return {'success': 'success', 'elapsed_time': elapsed_time}

@app.post('/push_html_1/')
async def loadHTML(html_docs: List[str] = Form(...), source_urls: List[str] = Form(...), hash: str = Form(...)):
    """Endpoint to load HTML content."""
    logging.info('Loading HTML')
    web_manager.push_html(hash, source_urls, html_docs)
    return {'success': 'success'}

@app.post('/delete_html/')
async def deleteHTML(hash: str = Form(...)):
    """Endpoint to delete HTML content."""
    logging.info('Deleting HTML')
    web_manager.delete_html(hash)
    return {'success': 'success'}

@app.post('/pull_memory/')
async def pullRelevantMemoriesForUser(query: str = Form(...), user_id: str = Form(...), context: str = Form(...), num_chunks: int = Form(...), num_neighbors: int= Form(...),similarity_threshold: float = Form(...)):
    logging.info(f'Pulling relevant memories for user {user_id}')
    memory_manager = MemoryManager(user_id, num_chunks)
    memories, elapsed_time = memory_manager.get_relevant_memory_docs(
        query, context=context, num_chunks=num_chunks, num_neighbors=num_neighbors, similarity_threshold=similarity_threshold)
    logging.info('Pulled relevant memories for user %s, query: %s, context: %s',
                 user_id, query, context)  # log the data pull
    logging.info('Elapsed time for operation: %s',
                 elapsed_time)  # log the elapsed time

    return {'memories': memories, 'elapsed_time': elapsed_time}

@app.post('/pull_memory_1/')
async def pullRelevantMemoriesForUser(query: str = Form(...), user_id: str = Form(...), context: str = Form(...), num_chunks: int = Form(...), num_neighbors: int= Form(...),similarity_threshold: float = Form(...)):
    """Endpoint to pull relevant memories for a specific user."""
    logging.info(f'Pulling relevant memories for user {user_id}')
    memories, elapsed_time = memory_manager1.pull_memory(user_id, query, context=context)
    logging.info(f'Pulled relevant memories for user {user_id}, query: {query}, context: {context}')  # log the data pull
    logging.info(f'Elapsed time for operation: {elapsed_time}')  # log the elapsed time

    return {'memories': memories, 'elapsed_time': elapsed_time}

@app.post('/semantic_search_html/')
async def semanticSearchHTML(query: str = Form(...), user_id: str = Form(...), context: str = Form(...), num_results: int = Form(...), similarity_threshold: float = Form(...)):
    logging.info(f'Semantic search HTML')
    memory_manager = MemoryManager(user_id, k_num=num_results)
    results, elapsed_time = memory_manager.semantic_search_html(
        query, context, similarity_threshold)
    logging.info('Pulled relevant results for query: %s, context: %s',
                 query, context)  # log the data pull
    logging.info('Elapsed time for operation: %s',
                 elapsed_time)  # log the elapsed time
    return {'results': results, 'elapsed_time': elapsed_time}

@app.post('/semantic_search_html_1/')
async def semanticSearchHTML(query: str = Form(...), hash: str = Form(...)):
    """Endpoint to conduct a semantic search in HTML content."""
    logging.info('Semantic search HTML')
    results, elapsed_time = memory_manager1.pull_html(hash, query)
    logging.info(f'Pulled relevant results for query: {query}')  # log the data pull
    logging.info(f'Elapsed time for operation: {elapsed_time}')  # log the elapsed time
    return {'results': results, 'elapsed_time': elapsed_time}

class ActionItem(BaseModel):
    action: str
    intent: str
    category: str

class FunctionInput(BaseModel):
    action_items: List[ActionItem] = Field(..., example=[{"action": "action_example", "intent": "intent_example", "category": "category_example"}])
    num_results: int = Field(..., example=5)
    similarity_threshold: float = Field(..., example=0.8)

@app.post('/get_functions/')
async def getFunctions(function_input: FunctionInput):
    action_items = function_input.action_items
    num_results = function_input.num_results
    similarity_threshold = function_input.similarity_threshold
    # try:
    memory_manager = MemoryManager("functions_test", k_num=num_results)
    
    logging.info(f'Processing Action Item: {action_items}')
    result, cb = await memory_manager.get_functions(action_items, num_results=num_results, similarity_threshold=similarity_threshold)
    
    logging.info('Pulled %i relevant results for query: %s', num_results, action_items)
    logging.info('Elapsed time for operation: %s', cb)
        
    return result
    # except Exception as e:
    #     logging.error(str(e))
    #     raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

@app.post('/get_functions_1/')
async def getFunctions(function_input: FunctionInput):
    """Endpoint to get functions based on provided input."""
    action_items = function_input.action_items

    logging.info(f'Processing Action Item: {action_items}')
    result, cb = await functions_manager1.pull_functions(action_items)
    logging.info(f'Pulled relevant results for query: {action_items}')
    logging.info(f'Elapsed time for operation: {cb}')

    return result

@app.post('/overwrite_functions/')
async def overwriteFunctions(functionsJson: str = Form(...), examplesJson: str = Form(...)):

    logging.info(f'Overwriting functions')

    with open('utils/functions.json', 'w') as f:
        f.write(functionsJson)
        f.close() 

    with open('utils/functions.json', 'r') as f:
        functionsJson = json.load(f)
        f.close()
    
    with open('utils/examples.json', 'w') as e:
        e.write(examplesJson)
        e.close()

    with open('utils/examples.json', 'r') as e:
        examplesJson = json.load(e)
        e.close()

    if functionsJson is None:
        return {'Reverted': True} 
    if functionsJson['informationretrieval_functions'] is None:
        return {'Reverted': True}

    functions_manager = FunctionsManager() 
    result = functions_manager.transform_and_push(functionsJson,examplesJson,"functions_test",mode=1)
    logging.info('Overwrote functions')

    return result

@app.post('/overwrite_functions_1/')
async def overwriteFunctions(functionsJson: str = Form(...)):
    """Endpoint to overwrite functions."""
    logging.info('Overwriting functions')
    with open('utils/functions.json', 'w') as f:
        f.write(functionsJson)
    with open('utils/functions.json', 'r') as f:
        functionsJson = json.load(f)

    if functionsJson is None or functionsJson['informationretrieval_functions'] is None:
        return {'Reverted': True} 

    result = functions_manager1.push_functions(functionsJson)
    logging.info('Overwrote functions')

    return result

@app.post('/clear_user_memory/')
async def clearUserMemory(user_id: str = Form(...)):
    logging.info(f'Clearing user memory for user {user_id}')
    memory_manager = MemoryManager(user_id)
    logging.info('Cleared user memory for user %s', user_id)
    return memory_manager.clear_user_memory()

@app.post('/clear_user_memory_1/')
async def clearUserMemory(user_id: str = Form(...)):
    """Endpoint to clear memory for a specific user."""
    logging.info(f'Clearing user memory for user {user_id}')
    memory_manager1.delete_memory(user_id)
    logging.info(f'Cleared user memory for user {user_id}')
    return {'success': 'success'}


@app.get('/test_callback/')
async def test_callback():
    """Test callback endpoint."""
    logging.info('Test callback')
    return {'test': 'test'}