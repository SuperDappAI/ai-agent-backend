import traceback
import logging
import random
import asyncio
import tiktoken

from datetime import datetime
from typing import List, Tuple
from langchain.schema import Document
from langchain.schema import SystemMessage, HumanMessage
from langchain.llms import OpenAI
from qdrant_retriever import MemoryType

INPUT_FACTOR = 0.6

class TreeSummaryPrompt:
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def to_system_message(self) -> SystemMessage:
        # Always assume multiple documents for simplification
        return SystemMessage(content=f"Summarize the two memory payloads together. Keep in mind their importance. The format is a pair of Memory payload: <payload> [importance: <low,medium,high>]")

    def to_human_message(self) -> HumanMessage:
        document_list_str = "\n".join([f"Memory payload: {doc.page_content} [importance: {doc.metadata['importance']}]" for doc in self.documents if doc is not None])
        return HumanMessage(content=document_list_str)

class FlexibleDocumentTreeSummarizer:
    _llm: OpenAI
    _verbose: bool
    _encoding: tiktoken.Encoding

    def __init__(self, llm: OpenAI, verbose: bool) -> None:
        self._llm = llm
        self._verbose = verbose
        self._encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.IMPORTANCE_SCORE_MAP = {
            "low": 1,
            "medium": 2,
            "high": 3
        }

    def importance_to_score(self, importance: str) -> int:
        return self.IMPORTANCE_SCORE_MAP.get(importance, 0)  # Default to 0 if invalid importance

    def score_to_importance(self, score: int) -> str:
        return {v: k for k, v in self.IMPORTANCE_SCORE_MAP.items()}.get(score, "invalid")


    def prepare_binary_summary(self, document_pair: List[Document]) -> TreeSummaryPrompt:
        total_importance = sum(self.importance_to_score(doc.metadata["importance"]) for doc in document_pair)
        average_importance = self.score_to_importance(round(total_importance / len(document_pair)))
        return TreeSummaryPrompt(documents=document_pair), average_importance

    async def _get_single_summary(self, summary_prompt: TreeSummaryPrompt, average_importance: str) -> Document:
        try:
            nowStamp = datetime.now().timestamp()
            messages = [[summary_prompt.to_system_message(), 
                        summary_prompt.to_human_message()]]
            response = await self._llm.agenerate(messages)
            if not response.generations or not response.generations[0]:
                raise ValueError("LLM did not provide a valid summary response.")
            summarized_content = response.generations[0][0].text
            metadata = {
                "id": random.randint(0, 2**32 - 1),
                "extra_index": summary_prompt.documents[0].metadata["extra_index"],
                "created_at": nowStamp,
                "importance": average_importance,
                "last_accessed_at": nowStamp,
                "summarizations": 100, # prevent further summarizations in future until more memories come in and its recalled
                "group_id": summary_prompt.documents[0].metadata["group_id"],
                "memory_type": MemoryType.SUBCONSCIOUS_MEMORY.value,
            }
            return Document(page_content=summarized_content,metadata=metadata)
        except Exception as e:
            #if self._verbose:
            logging.warn(f"FlexibleDocumentSummarizer: _get_single_summary exception e: {e}\n{traceback.format_exc()}")
        return None

    async def _get_layer_summaries_binary(self, documents: List[Document]) -> List[Document]:
        tasks = []
        odd_document = None
        for i in range(0, len(documents), 2): # Going in steps of 2
            # Pair up documents. If there's an odd one out, hold on to it.
            document_pair = documents[i:i+2]

            # If there's only one document in the pair, remember it for the next iteration
            if len(document_pair) == 1:
                odd_document = document_pair[0]
                continue

            summary_prompt, average_importance = self.prepare_binary_summary(document_pair)
            
            task = self._get_single_summary(summary_prompt, average_importance)
            tasks.append(task)

        # If there's an odd document remaining at the end of this layer, add it to the results
        # This will ensure it's combined with another document in the next layer
        summarized_docs = [doc for doc in await asyncio.gather(*tasks) if doc is not None]
        if odd_document:
            summarized_docs.append(odd_document)
            
        return summarized_docs

    async def tree_summarize(self, documents: List[Document], new_docs: List[Document] = None) -> List[Document]:
        if not documents:
            return []
        if len(documents) == 1:
            return documents
        if new_docs is None:
            new_docs = []
        layer_summaries = await self._get_layer_summaries_binary(documents)
        if not layer_summaries:
            return documents  # If no summaries are generated, return the original documents
        new_docs.extend(layer_summaries)
        return await self.tree_summarize(layer_summaries, new_docs)

    async def recursive_summary(self, documents: List[Document]) -> List[Document]:
        new_docs = await self.tree_summarize(documents)
        if len(new_docs) > 1:
            return await self.recursive_summary(new_docs)
        else:
            return [new_docs[0]]

    async def aupdate_documents(self, documents: List[Document]):
        return await self.recursive_summary(documents)