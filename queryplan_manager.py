import time
import logging
import os
from dotenv import load_dotenv
from langchain.schema import SystemMessage, ChatMessage
from langchain.chat_models import ChatOpenAI
import re


class QueryPlanManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")

        # gpt-4
        self.llm = ChatOpenAI(model='gpt-4', temperature=0)

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

        messages = [self.system_message, ChatMessage(
            content=str(query), role="user")]
        response = self.llm(messages)
        logging.info(response.content)
        end = time.time()
        logging.info(
            f"QueryPlanManager: query_plan operation took {end - start} seconds")
        return self.parse(response.content), {end - start}
