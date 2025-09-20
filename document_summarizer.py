import asyncio
import json
import logging
import traceback
from typing import Sequence

from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class SummaryPrompt:
    def __init__(self, summarizations: int, importance: str):
        self.importance = importance
        self.summarizations = summarizations

    def to_prompt_string(self) -> str:
        summarization_description = ""
        # Adjust the message based on the number of summarizations
        if self.summarizations == 1:
            summarization_description = "has already been summarized once."
        else:
            summarization_description = (
                f"has already been summarized {self.summarizations} times."
            )
        return f"Summarize this memory. Keep in mind it's of {self.importance} importance and {summarization_description}."


class FlexibleDocumentSummarizer:
    _llm: ChatOpenAI
    verbose: bool

    def __init__(self, llm: ChatOpenAI, verbose: bool = False) -> None:
        self._llm = llm
        self.verbose = verbose

    async def _get_single_summary(self, document: Document) -> None:
        try:
            summary_prompt = SummaryPrompt(
                summarizations=document.metadata["summarizations"],
                importance=document.metadata["importance"],
            )
            memory = json.loads(document.page_content)
            summary_prompt_str = summary_prompt.to_prompt_string()
            user_message = [
                SystemMessage(content=summary_prompt_str),
                HumanMessage(content=memory["user"]),
            ]
            aida_message = [
                SystemMessage(content=summary_prompt_str),
                HumanMessage(content=memory["AiDA"]),
            ]
            response = await self._llm.agenerate([user_message, aida_message])
            if (
                not response.generations
                or not response.generations[0]
                or not response.generations[1]
            ):
                raise Exception("LLM did not provide a valid summary response.")
            # Update the document's page_content in place with the summarized text
            document.page_content = json.dumps(
                {
                    "user": response.generations[0][0].text,
                    "AiDA": response.generations[1][0].text,
                }
            )
        except Exception as e:
            if self.verbose:
                logging.warning(
                    f"FlexibleDocumentSummarizer: _get_single_summary exception e: {e}\n{traceback.format_exc()}"
                )

    async def asummarize(self, documents: Sequence[Document]) -> None:
        tasks = [self._get_single_summary(document) for document in documents]
        await asyncio.gather(*tasks)
