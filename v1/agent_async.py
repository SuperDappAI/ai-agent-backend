from langchain.chat_models import ChatOpenAI 
from langchain import OpenAI 
from langchain.agents import Agent 
from langchain.agents import AgentType
# from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.callbacks import get_openai_callback
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.tools import Tool


async def call_async_agent(query):

    # weather = OpenWeatherMapAPIWrapper()

    gpt3 = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
    gpt4 = ChatOpenAI(model_name="gpt-4",temperature=0)
    davinci002 = OpenAI(model_name="text-davinci-002")
    davinci003 = OpenAI(model_name="text-davinci-003")
    # some other tools have no native support for async, so we'll test one that surely works out of the box.
    tools = load_tools(["llm-math"], llm=gpt4)

    agent = initialize_agent(llm=gpt4, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, tools=tools,verbose=False)

    with get_openai_callback() as cb:
        response = await agent.acall(query,return_only_outputs=True) 

    return response, cb