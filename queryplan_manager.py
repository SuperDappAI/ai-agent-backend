import time
import os
from dotenv import load_dotenv
from llama_index import ServiceContext, LLMPredictor
from llama_index.llms import OpenAI
from llama_index.indices.empty.base import EmptyIndex
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.response_synthesizers import Accumulate
from langchain.experimental.plan_and_execute import load_chat_planner
from langchain.schema import SystemMessage, ChatMessage
from langchain.chat_models import ChatOpenAI
import re



class QueryPlanManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        # LLMPredictor (gpt-4)
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4"))
        self.llm = ChatOpenAI(temperature=0)
        service_context = ServiceContext.from_defaults(llm=llm_predictor)
        # gpt-4
        step_decompose_transform = StepDecomposeQueryTransform(
            llm_predictor, verbose=True
        )
        index = EmptyIndex(service_context=service_context)
        query_engine = index.as_query_engine(
            service_context=service_context
        )
        self.query_engine = MultiStepQueryEngine(query_engine, query_transform=step_decompose_transform,response_synthesizer=Accumulate(service_context=service_context))
        self.system_message = SystemMessage(content="Let's first understand the problem and devise a plan to solve the problem."
        " Please output the plan starting with the header 'Plan:' "
        "and then followed by a numbered list of steps. "
        "Please make the plan the minimum number of steps required "
        "to accurately complete the task. If the task is a question, "
        "the final step should almost always be 'Given the above steps taken, "
        "please respond to the users original question'. "
        "At the end of your plan, say '<END_OF_PLAN>'")
    def parse(self, text: str):
        steps = [v for v in re.split("\n\s*\d+\. ", text)[1:]]
        return steps 

    def query_plan(self, query):
        start = time.time()

        messages = [self.system_message, ChatMessage(content=str(query),role="user")]
        response = self.llm(messages)
        print(response.content)
        # response = self.query_engine.query(query)
        end = time.time()
        print(f"QueryPlanManager: query_plan operation took {end - start} seconds")
        return self.parse(response.content), {end - start}
