import time
import logging
import re

from langchain.schema import SystemMessage, ChatMessage
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel

class QueryPlanInput(BaseModel):
    api_key: str
    query: str

    def __str__(self):
        return self.api_key + self.query

    def __eq__(self,other):
        return self.api_key == other.api_key and self.query == other.query

    def __hash__(self):
        return hash(str(self))

class QueryPlanManager:
    def __init__(self):
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

    def query_plan(self, query_input: QueryPlanInput):
        start = time.time()
        self.llm = ChatOpenAI(model='gpt-4', temperature=0, openai_api_key=query_input.api_key)
        messages = [self.system_message, ChatMessage(
            content=str(query_input.query), role="user")]
        response = self.llm(messages)
        logging.info(response.content)
        end = time.time()
        logging.info(
            f"QueryPlanManager: query_plan operation took {end - start} seconds")
        return self.parse(response.content), {end - start}
