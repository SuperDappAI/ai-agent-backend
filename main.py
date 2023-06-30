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
import asyncio
from streamingFunctionsAgent import streamingFunctionsAgent
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
]

app = FastAPI()

# run with uvicorn main:app --reload

pinecone.init()

@app.post('/push_memory/')
async def writeMemoryForUser(message: str = Form(...), llm_response: str = Form(...), user_id: str = Form(...)):
    memory_manager = MemoryManager(user_id)
    elapsed_time = memory_manager.push_memory(message,llm_response)
    return {'elapsed_time': elapsed_time} 

@app.post('/pull_memory/')
async def pullRelevantMemoriesForUser(query: str = Form(...),user_id: str = Form(...),context: str = Form(...),k_num: int = Form(...),deepSearch: bool = Form(...)):
    memory_manager = MemoryManager(user_id,k_num)
    memories, elapsed_time = memory_manager.get_relevant_memory_docs(query,deepSearch=deepSearch, context=context) 
    return memories, elapsed_time

@app.post('/clear_user_memory/')
async def clearUserMemory(user_id: str = Form(...)):
    memory_manager = MemoryManager(user_id)
    return memory_manager.clear_user_memory()