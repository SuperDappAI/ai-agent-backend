from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os

from agent import call_agent
from curie_async import call_curie_async 
from agent_async import call_async_agent 
from agent_async2 import call_async_agent2 


from langchain.chat_models import ChatOpenAI 
from langchain import OpenAI

import asyncio

load_dotenv()
os.getenv('OPENAI_API_KEY')
os.getenv('SERPER_API_KEY')

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://localhost:8000",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# run with uvicorn main:app --reload

@app.post('/query/')
async def callAgent(query: str = Form(...)):
    gptAnswer,cb = call_agent(query)
    result = {'query': query, 'answer': gptAnswer, 'OpenAI_callback': cb}
    return result

@app.post('/query_async_agent/')
async def callAsyncAgent(query: str = Form(...)):
    gptAnswer,cb = await call_async_agent(query)
    result = {'answer': gptAnswer, 'OpenAI_callback': cb}
    return result

@app.post('/query_curie/')
async def call_async_curie(query: str = Form(...)):
    result = await call_curie_async(query)
    return result 

@app.post('/query_async_agent2/')
async def callAsyncAgent(query: str = Form(...)):
    response,cb = await call_async_agent2(query)
    gptAnswer = response
    result = {'answer': gptAnswer, 'OpenAI_callback': cb}
    return result

@app.post('/query_async_agent_with_memory/')
async def callAsyncAgent(query: str = Form(...), username: str = Form(...)):
    response,cb = await call_async_agent_with_memory(query, username)
    gptAnswer = response
    result = {'answer': gptAnswer, 'OpenAI_callback': cb}
    return result
