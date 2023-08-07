import asyncio
from typing import Sequence
from datetime import datetime
from math import ceil
from langchain.schema import Document
from pydantic import Field
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

class SummaryPrompt:
    def __init__(self, flexibility_score: int):
        self.flexibility_score = min(max(flexibility_score, 0), 10)

    def to_prompt_string(self) -> str:
        return (f"Summarize this text with a flexibility score of {self.flexibility_score}. "
                "A score of 10 means no loss of detail, only rewording or reorganizing for clarity. "
                "A score of 0 allows full summarization flexibility while retaining the general context.")

class FlexibleDocumentSummarizer:
    _llm: ChatOpenAI
    """The model used to summarize."""
    
    _decay_rate: float = Field(default=0.0314)
    """The decay factor going into the power law for forgetting."""

    _use_async: bool
    """Determines if the summarization should be processed asynchronously."""

    _verbose: bool
    """Controls the verbosity of logging during the summarization process."""

    def __init__(self, llm: ChatOpenAI, decay_rate: float = 0.0314, use_async: bool = False, verbose: bool = False) -> None:
        self._llm = llm
        self._decay_rate = decay_rate
        self._use_async = use_async
        self._verbose = verbose

    def _calculate_flexibility_score(self, current_time: datetime, document: Document) -> int:
        e = int(document.metadata["summarizations"])
        I = float(document.metadata.get('importance_score', 0.0)) / 10.0
        importance_score = self._power_law_forgetting(I, e)
        return round(importance_score * 10)

    def _power_law_forgetting(self, I, e):
        if e == 0:
            return I
        return (1 - self._decay_rate * e)

    async def _get_single_summary(self,  document: Document) -> None:
        flexibility_score = self._calculate_flexibility_score(datetime.now(), document)
        summary_prompt = SummaryPrompt(flexibility_score=flexibility_score)

        text_chunk = document.page_content
        messages = [[SystemMessage(content=summary_prompt.to_prompt_string()), 
                    HumanMessage(content=str(text_chunk))]]
        response = await self._llm.agenerate(messages)

        summarized_content = response.generations[0][0].text
        # Update the document's page_content in place with the summarized text
        document.page_content = summarized_content


    async def aupdate_documents(self,  documents: Sequence[Document]) -> None:
        tasks = [self._get_single_summary(document) for document in documents]
        await asyncio.gather(*tasks)
