
from langchain.chat_models import ChatOpenAI 
from langchain import OpenAI 
from langchain.agents import Agent 
from langchain.agents import AgentType
# from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks import get_openai_callback
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.tools import Tool


def call_agent(query):

    # print(weather)

    # weather = OpenWeatherMapAPIWrapper()

    gpt3 = ChatOpenAI(model_name="gpt-3.5-turbo")
    gpt4 = ChatOpenAI(model_name="gpt-4",temperature=0.2)
    davinci002 = OpenAI(model_name="text-davinci-002")
    davinci003 = OpenAI(model_name="text-davinci-003")

    # weather_data = weather.run("London, GB")
    # print(weather_data)

    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]
    agent = initialize_agent(llm=gpt4, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, tools=tools,verbose=True)

    with get_openai_callback() as cb:
        response = agent.run(query)
        agent.run()
        print(cb)

    return response, cb