import time
import logging
import asyncio
import traceback

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.schema import SystemMessage, HumanMessage
from preferences_resolver import PreferencesResolver
from classify_prompts import ClassifyPrompts


class QueryPlanInput(BaseModel):
    api_key: str
    query: str
    conversation_id: str


class QueryPlanManager:
    classifyPrompts: ClassifyPrompts

    def __init__(self):
        self.classify_prompts = ClassifyPrompts()

    async def query_plan(self, preferences_resolver: PreferencesResolver, query_input: QueryPlanInput):
        start = time.time()
        roleDB = await preferences_resolver.get_role(query_input.conversation_id)
        if roleDB is None:
            try:
                messages = [[SystemMessage(content=self.classify_prompts.to_prompt_string()),
                             HumanMessage(content=query_input.query)]]
                llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0,
                                 max_tokens=8, openai_api_key=query_input.api_key)
                response = await llm.agenerate(messages)
                if not response.generations or not response.generations[0]:
                    raise Exception(
                        "LLM did not provide a valid summary response.")
                result = response.generations[0][0].text
                role = self.classify_prompts.parseClassification(result)
                if role is None:
                    end = time.time()
                    return "No plan needed", {end - start}
                asyncio.create_task(preferences_resolver.set_role(
                    result, query_input.conversation_id))
            except Exception as e:
                logging.warning(
                    f"QueryPlanManager: query_plan exception, e: {e}\n{traceback.format_exc()}")
                end = time.time()
                return "No plan needed", {end - start}
        else:
            role = self.classify_prompts.parseClassification(roleDB)
            if role is None:
                end = time.time()
                return "No plan needed", {end - start}
        end = time.time()
        logging.info(
            f"QueryPlanManager: query_plan operation took {end - start} seconds")
        return role, {end - start}
