# from langchain.chat_models import ChatOpenAI 
from langchain import OpenAI 
# from langchain.callbacks import get_openai_callback
# import asyncio

curie = OpenAI(model_name="text-curie-001")

async def call_curie_async(query):
    result = await curie._call_async(query)
    return result
