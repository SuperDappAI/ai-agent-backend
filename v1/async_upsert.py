from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os

import asyncio
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from async_memory import VectorStoreRetrieverMemoryBuffer 

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

pinecone.init()
history_db = Pinecone.from_existing_index(index_name="chat-message-history",embedding=OpenAIEmbeddings())
retriever = history_db.as_retriever()
memory = VectorStoreRetrieverMemory(retriever=retriever,input_key="chat_history")
async_memory = VectorStoreRetrieverMemoryBuffer(retriever=retriever,input_key="chat_history")

async def upsert_query_serial(query):
    return memory.save_context({"chat_history": query},{"output": query})

# async def upsert_query_parallel(query):
    # return async_memory.save_context({"chat_history": query},{"output": query}) 
    # return True

# run with uvicorn main:app --reload

@app.post('/query/')
async def upsert_query(query: str = Form(...)):
    await upsert_query_serial(query)
    return True 

@app.post('/query_async/')
async def upsert_query_async(query: str = Form(...)):
    await async_memory.save_context({"chat_history": query},{"output": query})
    # await upsert_query_parallel(query)
    return True 

