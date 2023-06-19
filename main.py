from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os
import pinecone

from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from memory import MemoryManager

import asyncio

load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("SERPER_API_KEY")

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://localhost:8000",
]

app = FastAPI()

# run with uvicorn main:app --reload


@app.post("/query/")
async def callAgent(query: str = Form(...)):
    gptAnswer, cb = instance_agent_with_memory(query)
    result = {"query": query, "answer": gptAnswer, "OpenAI_callback": cb}
    return result


# @app.post('/query_async_agent/')
# async def callAsyncAgent(query: str = Form(...)):
#     gptAnswer,cb = await call_async_agent(query)
#     result = {'answer': gptAnswer, 'OpenAI_callback': cb}
#     return result

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI 
from langchain import OpenAI

import asyncio


load_dotenv()
os.getenv('OPENAI_API_KEY')
# os.getenv('SERPER_API_KEY')

pinecone.init()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://localhost:8000",
]

app = FastAPI()

# figure out middleware later
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# run with uvicorn main:app --reload


@app.post('/push_memory/')
async def writeMemoryForUser(message: str = Form(...), llm_response: str = Form(...), user_id: str = Form(...)):
    memory_manager = MemoryManager(user_id)
    elapsed_time = memory_manager.push_memory(message,llm_response)
    return {'elapsed_time': elapsed_time} 

@app.post('/pull_memory/')
async def pullRelevantMemoriesForUser(query: str = Form(...),user_id: str = Form(...)):
    memory_manager = MemoryManager(user_id)
    memories,elapsed_time = memory_manager.get_relevant_memories(query) 
    result = {'chat_history': memories, 'elapsed_time': elapsed_time}
    return result

@app.post('/clear_user_memory/')
async def clearUserMemory(user_id: str = Form(...)):
    memory_manager = MemoryManager(user_id)
    return memory_manager.clear_user_memory()