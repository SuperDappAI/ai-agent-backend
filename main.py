from fastapi import FastAPI, Form, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware

import openai
from dotenv import load_dotenv
import os
import pinecone

from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from memory_search import MemoryManager

from aiohttp import ClientSession
import logging
# import asyncio
# from streamingFunctionsAgent import streamingFunctionsAgent
# from sse_starlette import EventSourceResponse


load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("SERPER_API_KEY")
os.getenv("PINECONE_API_KEY")

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://localhost:8000",
    "https://python-api.chatdapp.dev",
]

app = FastAPI()

# run with uvicorn main:app --reload

pinecone.init()

LOGFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log') 
logging.basicConfig(filename=LOGFILE_PATH, filemode='a',format='%(name)s - %(message)s',force=True)

@app.post('/push_memory/')
async def writeMemoryForUser(message: str = Form(...), llm_response: str = Form(...), user_id: str = Form(...)):
    logging.info(f'Writing memory for user {user_id}')
    memory_manager = MemoryManager(user_id)
    elapsed_time = memory_manager.push_memory(message, llm_response)
    logging.info('Pushed memory for user %s, message: %s, response: %s', user_id, message, llm_response)  # log the data push
    logging.info('Elapsed time for operation: %s', elapsed_time)  # log the elapsed time

    return {'elapsed_time': elapsed_time}

@app.post('/pull_memory/')
async def pullRelevantMemoriesForUser(query: str = Form(...), user_id: str = Form(...), context: str = Form(...), k_num: int = Form(...), deepSearch: bool = Form(...)):
    logging.info(f'Pulling relevant memories for user {user_id}')
    memory_manager = MemoryManager(user_id, k_num)
    memories, elapsed_time = memory_manager.get_relevant_memory_docs(
        query, deepSearch=deepSearch, context=context)
    logging.info('Pulled relevant memories for user %s, query: %s, context: %s', user_id, query, context)  # log the data pull
    logging.info('Elapsed time for operation: %s', elapsed_time)  # log the elapsed time

    return {'memories': memories, 'elapsed_time': elapsed_time}

@app.post('/clear_user_memory/')
async def clearUserMemory(user_id: str = Form(...)):
    logging.info(f'Clearing user memory for user {user_id}')
    memory_manager = MemoryManager(user_id)
    logging.info('Cleared user memory for user %s', user_id)
    return memory_manager.clear_user_memory()

@app.get('/test_callback/')
async def test_callback():
    logging.info(f'Test callback')

    return {'test': 'test'}