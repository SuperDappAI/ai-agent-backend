import asyncio
import traceback
import logging
from typing import Sequence
from langchain.schema import Document
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

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
            summarization_description = f"has already been summarized {self.summarizations} times."
        return (f"Summarize this memory. Keep in mind it's of {self.importance} importance and {summarization_description}")

class FlexibleDocumentSummarizer:
    _llm: ChatOpenAI
    _verbose: bool

    def __init__(self, llm: ChatOpenAI, verbose: bool = False) -> None:
        self._llm = llm
        self._verbose = verbose

    async def _get_single_summary(self,  document: Document) -> None:
        try:
            summary_prompt = SummaryPrompt(summarizations=document.metadata["summarizations"],importance=document.metadata["importance"])
            text_chunk = document.page_content

            messages = [[SystemMessage(content=summary_prompt.to_prompt_string()), 
                        HumanMessage(content=str(text_chunk))]]

            response = await self._llm.agenerate(messages)

            if not response.generations or not response.generations[0]:
                raise Exception("LLM did not provide a valid summary response.")

            summarized_content = response.generations[0][0].text

            # Update the document's page_content in place with the summarized text
            document.page_content = summarized_content

        except Exception as e:
            if self._verbose:
                logging.warn(f"FlexibleDocumentSummarizer: _get_single_summary exception on document: {document.id} e: {e}\n{traceback.format_exc()}")

    async def asummarize(self,  documents: Sequence[Document]) -> None:
        tasks = [self._get_single_summary(document) for document in documents]
        await asyncio.gather(*tasks)
