import os
import logging
from dotenv import load_dotenv
import pinecone
from fastapi import FastAPI, Form
import json
from memory_search import MemoryManager
from agent_manager import AgentManager
from web_manager import WebManager, HTMLInput
from custom_text_loader import TextLoader
from functions_endpoint import FunctionsManager
from functions_manager import FunctionsManager1, FunctionInput
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
agent_manager = AgentManager()
web_manager = WebManager()
queryplan_manager = QueryPlanManager()

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutdown")
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
async def writeMemoryForUser(query: str = Form(...), llm_response: str = Form(...), user_id: str = Form(...), conversation_id: str = Form(...)):
    """Endpoint to push memory for a specific user."""
    logging.info(f'Writing memory for user {user_id}, conversation {conversation_id}')
    elapsed_time = agent_manager.push_memory(user_id, conversation_id, query, llm_response)
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

@app.post('/delete_html/')
async def deleteHTML(hash: str = Form(...)):
    """Endpoint to delete HTML content."""
    logging.info('Deleting HTML')
    elapsed_time = web_manager.delete_html(hash)
    return {'elapsed_time': elapsed_time}

@app.post('/pull_memory/')
async def pullRelevantMemoriesForUser(query: str = Form(...), user_id: str = Form(...), context: str = Form(...), num_chunks: int = Form(...), num_neighbors: int= Form(...),similarity_threshold: float = Form(...)):
    logging.info(f'Pulling relevant memories for user {user_id}')
    memory_manager = MemoryManager(user_id, num_chunks)
    try:
        memories, elapsed_time = memory_manager.get_relevant_memory_docs(
        query, context=context, num_chunks=num_chunks, num_neighbors=num_neighbors, similarity_threshold=similarity_threshold)
    except:
        return {'memories': [], 'elapsed_time': 0, 'error': 'No memories found'}
    logging.info('Pulled relevant memories for user %s, query: %s, context: %s',
                 user_id, query, context)  # log the data pull
    logging.info('Elapsed time for operation: %s',
                 elapsed_time)  # log the elapsed time

    return {'memories': memories, 'elapsed_time': elapsed_time}

@app.post('/pull_memory_1/')
async def pullRelevantMemoriesForUser(query: str = Form(...), user_id: str = Form(...), conversation_id: str = Form(...)):
    """Endpoint to pull relevant memories for a specific user."""
    logging.info(f'Pulling relevant memories for user {user_id}, conversation {conversation_id}')
    memories, elapsed_time = agent_manager.pull_memory(user_id, conversation_id, query)
    return {'response': memories, 'elapsed_time': elapsed_time}

@app.post('/get_latest_memories/')
async def pullLatestMemoriesForUser(user_id: str = Form(...), token_count: int = Form(None)):
    """Endpoint to pull latest memories for a specific user based on token_count."""
    logging.info(f'Pulling latest memories for user {user_id}')
    memories, elapsed_time = agent_manager.get_latest_memories(user_id, token_count)
    return {'response': memories, 'elapsed_time': elapsed_time}

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
async def semanticSearchHTML(function_input: HTMLInput):
    """Endpoint to conduct a semantic search in HTML content."""
    logging.info('Semantic search HTML')
    results, elapsed_time = web_manager.search_html(function_input)
    return {'response': results, 'elapsed_time': elapsed_time}

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
    logging.info(f'Processing Action Item: {function_input.action_items}')
    result, elapsed_time = functions_manager1.pull_functions(function_input)
    return {'response': result, 'elapsed_time': elapsed_time}

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
    if functionsJson['information_retrieval'] is None:
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

    if functionsJson is None or functionsJson['information_retrieval'] is None:
        return {'Reverted': True} 

    result, elapsed_time = functions_manager1.push_functions(functionsJson)
    logging.info('Overwrote functions')

    return {'response': result, 'elapsed_time': elapsed_time}

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
    elapsed_time = agent_manager.delete_memory(user_id)
    return {'elapsed_time': elapsed_time}


@app.get('/test_callback/')
async def test_callback():
    """Test callback endpoint."""
    logging.info('Test callback')
    return {'test': 'test'}