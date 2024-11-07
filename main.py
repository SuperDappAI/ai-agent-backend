import os
import logging
import time
import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from agent_manager import AgentManager, MemoryInput, MemoryOutput, ClearMemory
from web_manager import WebManager, HTMLInput, CacheHTML
from doc_manager import DocManager, DocAddInput, DocDeleteInput, DocSearchInput, CacheDoc
from functions_manager import FunctionsManager, FunctionInput, FunctionOutput
from agents_manager import AgentsManager, AgentListInput, AgentPublishInput, AgentUnpublishInput, ClearAgentMemory, AgentMessageInput, AgentRegisterInput, AgentRegisterGroupInput
from queryplan_manager import QueryPlanManager, QueryPlanInput
from cache_manager import CacheClearInput
from preferences_resolver import QueryPreferencesInput
from cachetools import TTLCache, LRUCache
from rate_limiter import RateLimiter, SyncRateLimiter
from websocket_connection_manager import WebSocketConnectionManager

rate_limiter = RateLimiter(rate=5, period=1)  # Allow 5 tasks per second
rate_limiter_sync = SyncRateLimiter(rate=5, period=1)
# Load environment variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
CONSOLE_KEY = os.getenv("CONSOLE_KEY")

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
                    format='%(asctime)s.%(msecs)03d %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True, level=logging.INFO)


functions_manager = FunctionsManager(rate_limiter, rate_limiter_sync)
agent_manager = AgentManager(rate_limiter, rate_limiter_sync)
web_manager = WebManager(rate_limiter, rate_limiter_sync)
queryplan_manager = QueryPlanManager()
doc_manager = DocManager(rate_limiter, rate_limiter_sync)

queryplancache = TTLCache(maxsize=16384, ttl=36000)
searchhtmlcache = TTLCache(maxsize=16384, ttl=36000)
functioncache = TTLCache(maxsize=16384, ttl=36000)
agentcache = TTLCache(maxsize=16384, ttl=36000)
doccache = LRUCache(maxsize=16384)
active_connections = []
active_connections_lock = asyncio.Lock()

websocket_manager = WebSocketConnectionManager(
    active_connections=active_connections,
    active_connections_lock=active_connections_lock,
)

class MessageManager:
    """
    This class handles the automated generation and management of chat interactions
    using an automated workflow configuration and message queue.
    """

    def __init__(
        self, websocket_manager: WebSocketConnectionManager = None, human_input_timeout: int = 180
    ) -> None:
        """
        Initializes the AutoGenChatManager with a message queue.

        """
        self.websocket_manager = websocket_manager
        self.a_human_input_timeout = human_input_timeout

    async def a_send(self, message: dict) -> None:
        """
        Asynchronously sends a message via the WebSocketManager class

        :param message: The message string to be sent.
        """
        for connection, socket_client_id in self.websocket_manager.active_connections:
            if message["connection_id"] == socket_client_id:
                logging.info(
                    f"Sending message to client connection_id: {message['connection_id']}. Connection ID: {socket_client_id}"
                )
                await self.websocket_manager.send_message(message, connection)
            else:
                logging.info(
                    f"Skipping message for client connection_id: {message['connection_id']}. Connection ID: {socket_client_id}"
                )
    
    async def a_prompt_for_input(self, prompt: dict, timeout: int = 60) -> str:
        """
        Sends the user a prompt and waits for a response asynchronously via the WebSocketManager class

        :param message: The message string to be sent.
        """

        for connection, socket_client_id in self.websocket_manager.active_connections:
            if prompt["connection_id"] == socket_client_id:
                logging.info(
                    f"Sending message to client connection_id: {prompt['connection_id']}. Connection ID: {socket_client_id}"
                )
                try:
                    result = await self.websocket_manager.get_input(prompt, connection, timeout)
                    return result
                except Exception as e:
                    return f"Error: {e}\nTERMINATE"
            else:
                logging.info(
                    f"Skipping message for client connection_id: {prompt['connection_id']}. Connection ID: {socket_client_id}"
                )

message_manager = MessageManager(websocket_manager)
agents_manager = AgentsManager(rate_limiter, message_manager)
@app.post('/get_preferences/')
async def getPreferences(preferences_query: QueryPreferencesInput):
    logging.info(f'Get preferences for user {preferences_query.user_id}')
    start = time.time()
    response = await agent_manager.preferences_resolver.get_preferences(preferences_query.user_id)
    if response is None:
        logging.info(f'Preferences for user {preferences_query.user_id} does not exist, returnng default and making one in the background...')
        asyncio.create_task(agent_manager.preferences_resolver.create_default_preferences(preferences_query.user_id))
        response = agent_manager.preferences_resolver.default_preferences
    end = time.time()
    return {'response': response, 'elapsed_time': end - start}

@app.post('/query_plan/')
async def writeQueryPlan(query_input: QueryPlanInput):
    result = queryplancache.get(query_input.conversation_id)
    if result is not None:
        return {'response': result, 'elapsed_time': 0}
    logging.info(f'Writing query plan for conversation_id {query_input.conversation_id}')
    response, elapsed_time = await queryplan_manager.query_plan(agent_manager.preferences_resolver, query_input)
    logging.info('Elapsed time for operation: %s',
                 elapsed_time)  # log the elapsed time
    if response != "No plan needed":
        queryplancache[query_input.conversation_id] = response
    return {'response': response, 'elapsed_time': elapsed_time}

@app.post('/push_memory/')
async def writeMemoryForUser(memory_output: MemoryOutput):
    """Endpoint to push memory for a specific user."""
    logging.info(f'Writing memory for user for user {memory_output.user_id}, conversation {memory_output.conversation_id}')
    elapsed_time = await agent_manager.push_memory(memory_output)
    return {'elapsed_time': elapsed_time}

@app.post('/pull_memory/')
async def pullRelevantMemoriesForUser(memory_input: MemoryInput):
    """Endpoint to pull relevant memories for a specific user."""
    logging.info(f'Pulling relevant memories for user {memory_input.user_id}, conversation {memory_input.conversation_id}')
    memories, elapsed_time = await agent_manager.pull_memory(memory_input)
    return {'response': memories, 'elapsed_time': elapsed_time}

@app.post('/semantic_search_html/')
async def semanticSearchHTML(function_input: HTMLInput):
    """Endpoint to conduct a semantic search in HTML content."""
    result = searchhtmlcache.get(function_input)
    if result is not None:
        return {'response': result, 'elapsed_time': 0}
    logging.info('Semantic search HTML')
    results, elapsed_time = await web_manager.search_html(function_input)
    if len(results) > 0:
        searchhtmlcache[function_input] = results
    return {'response': results, 'elapsed_time': elapsed_time}

@app.post('/is_html_search_cached/')
async def isHTMLSearchCached(cache_html: CacheHTML):
    """Endpoint to check if HTML content is cached."""
    logging.info('Checking if HTML results are cached')
    result, elapsed_time = web_manager.does_hash_exist(cache_html.hash)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/add_doc/')
async def addDoc(function_input: DocAddInput):
    """Endpoint to conduct add HTML document for doc portal."""
    logging.info('add to Doc Portal')
    results, elapsed_time = await doc_manager.add_doc(function_input)
    doccache.clear()
    return {'response': results, 'elapsed_time': elapsed_time}

@app.post('/delete_doc/')
async def deleteDoc(function_input: DocDeleteInput):
    """Endpoint to conduct delete HTML document from doc portal."""
    logging.info('delete from Doc Portal')
    results, elapsed_time = doc_manager.delete_doc(function_input)
    doccache.clear()
    return {'response': results, 'elapsed_time': elapsed_time}

@app.post('/is_doc_cached/')
async def isDocCached(cache_html: CacheDoc):
    """Endpoint to check if doc content is cached."""
    logging.info('Checking if doc is cached')
    result, elapsed_time = doc_manager.does_source_exist(cache_html.source_url)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/search_doc/')
async def semanticSearchDoc(function_input: DocSearchInput):
    """Endpoint to conduct a semantic search in doc portal."""
    result = doccache.get(function_input)
    if result is not None:
        return {'response': result, 'elapsed_time': 0}
    logging.info('Semantic search Doc Portal')
    results, elapsed_time = await doc_manager.search_doc(function_input)
    if len(results) > 0:
        doccache[function_input] = results
    return {'response': results, 'elapsed_time': elapsed_time}

@app.post('/get_functions/')
async def getFunctions(function_input: FunctionInput):
    """Endpoint to get functions based on provided input."""
    result = functioncache.get(function_input)
    if result is not None:
        logging.info(f'Found functions in cache, result {result}')
        return {'response': result, 'elapsed_time': 0}
    logging.info(f'Processing Action Item: {function_input.action_items}')
    result, elapsed_time = await functions_manager.pull_functions(function_input)
    if len(result) > 0:
        functioncache[function_input] = result
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/list_registered_agents/')
async def listRegisteredAgents(agent_input: AgentListInput):
    """Endpoint to get registered agents based on provided input."""
    result = agentcache.get(agent_input)
    if result is not None:
        logging.info(f'Found agents in cache, result {result}')
        return {'response': result, 'elapsed_time': 0}
    result, elapsed_time = await agents_manager.list_registered_agents(agent_input)
    if len(result) > 0:
        agentcache[agent_input] = result
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/register_agent/')
async def registerAgent(agent_input: AgentRegisterInput):
    """Endpoint to register agent based on provided input."""
    logging.info(f'Registering agent: {agent_input}')
    result, elapsed_time = await agents_manager.register_agent(agent_input)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/unregister_agent/')
async def unregisterAgent(agent_input: AgentRegisterInput):
    """Endpoint to unregister agent based on provided input."""
    logging.info(f'Unregistering agent: {agent_input}')
    result, elapsed_time = await agents_manager.unregister_agent(agent_input)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/add_agent_to_conversation/')
async def addAgentToConveration(agent_input: AgentRegisterGroupInput):
    """Endpoint to add agent to conversation based on provided input."""
    logging.info(f'Adding agent to conversation: {agent_input}')
    result, elapsed_time = await agents_manager.add_agent_to_conversation(agent_input)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/add_agent_to_conversation/')
async def removeAgentFromConveration(agent_input: AgentRegisterGroupInput):
    """Endpoint to remove agent from conversation based on provided input."""
    logging.info(f'Removing agent from conversation: {agent_input}')
    result, elapsed_time = await agents_manager.remove_agent_from_conversation(agent_input)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/push_functions/')
async def pushFunctions(function_output: FunctionOutput):
    """Endpoint to push functions based on provided functions."""
    logging.info(f'Adding functions: {function_output.functions}')
    functions = {}
    function_types = ['information_retrieval', 'communication', 'data_processing', 'sensory_perception']

    for function_item in function_output.functions:
        function_item.category = function_item.category.lower().replace(' ', '_')
        if function_item.category not in function_types:
            return {'response': f'Invalid category for function {function_item.name}, must be one of {function_types}'}

        # Initialize category list if not already done
        if function_item.category not in functions:
            functions[function_item.category] = []

        # Append the new function to the category
        new_function = {
            'name': function_item.name,
            'description': function_item.description
        }

        functions[function_item.category].append(new_function)

    # Push the functions
    result, elapsed_time = await functions_manager.push_functions(function_output.user_id, function_output.api_key, functions)
    return {'response': result, 'elapsed_time': elapsed_time}


@app.post('/publish_agent/')
async def publishAgent(agent_output: AgentPublishInput):
    """Endpoint to push agents based on provided agents."""
    logging.info(f'Publishing agent: {agent_output}')
    result, elapsed_time = await agents_manager.publish_agent(agent_output)
    return {'response': result, 'elapsed_time': elapsed_time}

@app.post('/unpublish_agent/')
async def unpublishAgent(agent_input: AgentUnpublishInput):
    """Endpoint to push agents based on provided agents."""
    logging.info(f'Unpublishing agent: {agent_input}')
    result, elapsed_time = await agents_manager.unpublish_agent(agent_input)
    return {'response': result, 'elapsed_time': elapsed_time}


@app.post('/clear_conversation/')
async def clearUserMemory(clear_memory: ClearMemory):
    """Endpoint to clear memory for a specific user/conversation."""
    logging.info(
        f'Clearing user memory for user {clear_memory.user_id} and conversation {clear_memory.conversation_id}')
    response, elapsed_time = agent_manager.clear_conversation(clear_memory)
    return {'response': response, 'elapsed_time': elapsed_time}

@app.post('/clear_agent_conversation/')
async def clearAgentMemory(clear_memory: ClearAgentMemory):
    """Endpoint to clear memory for a specific agent conversation."""
    logging.info(
        f'Clearing agent memory for user {clear_memory.user_id} and conversation {clear_memory.conversation_id}')
    response, elapsed_time = agents_manager.clear_conversation(clear_memory)
    return {'response': response, 'elapsed_time': elapsed_time}

@app.post('/cache_clear/')
async def clearCache(cache_clear_input: CacheClearInput):
    """Endpoint to clear caches."""
    start = time.time()
    if not cache_clear_input.console_key.strip():
        logging.warn("CacheManager: console key is empty, check settings!")
        return {'response': "fail", 'elapsed_time': 0}
    if CONSOLE_KEY != cache_clear_input.console_key:
        logging.warn("CacheManager: Invalid console key")
        return {'response': "fail", 'elapsed_time': 0}
    if {"doc", "all"} & set(cache_clear_input.cache_types):
        doccache.clear()
    if {"queryplan", "all"} & set(cache_clear_input.cache_types):
        queryplancache.clear()
    if {"searchhtml", "all"} & set(cache_clear_input.cache_types):
        searchhtmlcache.clear()
    if {"agent", "all"} & set(cache_clear_input.cache_types):
        agentcache.clear()
    if {"function", "all"} & set(cache_clear_input.cache_types):
        functioncache.clear()
    end = time.time()
    return {'response': "success", 'elapsed_time': end - start}

@app.websocket("/ws/{client_id}")
async def messageAgent(websocket: WebSocket, client_id: str):
    """Endpoint to facilitate communication between the client and the agent."""
    await websocket_manager.connect(websocket, client_id)
    try:
        # Receive the initial message from the client
        data = await websocket.receive_json()
        
        # Create AgentMessageInput from the received data using **
        agent_input = AgentMessageInput(**data)
        
        # Validate the client_id matches the user_id in agent_input
        if client_id != agent_input.user_id:
            # Send an error message over the WebSocket connection
            await websocket.send_json({
                'type': 'error',
                'message': "Error: connection_id does not match client_id",
            })
            await websocket_manager.disconnect(client_id)
            return
        
        logging.info(f'Messaging agent: {agent_input}')
        
        # Start the message processing
        await agents_manager.message_agent(agent_input)
        
    except WebSocketDisconnect:
        logging.info(f"Client #{client_id} disconnected")
        await websocket_manager.disconnect(client_id)
    except Exception as e:
        logging.error(f"Error in messageAgent: {e}")
        await websocket.send_json({'type': 'error', 'message': str(e)})
        await websocket_manager.disconnect(client_id)
