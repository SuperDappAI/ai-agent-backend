from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from routerAgent import call_agent

from langchain.chat_models import ChatOpenAI 
from langchain import OpenAI


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

# @app.post('/chat/')
# async def chatWithAgent(chat: str = Form(...)):
