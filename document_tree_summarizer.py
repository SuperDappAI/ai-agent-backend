import traceback
import logging
import random
import asyncio
import tiktoken

from datetime import datetime
from typing import List, Tuple
from langchain.schema import Document
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from qdrant_retriever import MemoryType

class TreeSummaryPrompt:
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def to_system_message(self) -> SystemMessage:
        if len(self.documents) == 1:
            return SystemMessage(content="Summarize this memory more concisely. Keep in mind its importance: [importance: <low,medium,high>]")
        else:
            return SystemMessage(content=f"Summarize these {len(self.documents)} memories. Keep in mind their importance. The format is a list of Memory: <memory content> [importance: <low,medium,high>]")

    def to_human_message(self) -> HumanMessage:
        document_list_str = "\n".join([f"Memory: {doc.page_content} [importance: {doc.metadata['importance']}]" for doc in self.documents if doc is not None])
        return HumanMessage(content=document_list_str)

class FlexibleDocumentTreeSummarizer:
    _llm: ChatOpenAI
    _verbose: bool
    _input_limit: int
    _encoding: tiktoken.Encoding

    def __init__(self, llm: ChatOpenAI, verbose: bool = False) -> None:
        self._llm = llm
        self._verbose = verbose
        self._input_limit = 1000 * 0.6
        self._encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")

    def importance_to_score(self, importance: str) -> int:
        if importance == "low":
            return 1
        elif importance == "medium":
            return 2
        elif importance == "high":
            return 3
        else:
            raise ValueError(f"Invalid importance level: {importance}")

    def score_to_importance(self, score: int) -> str:
        if score == 1:
            return "low"
        elif score == 2:
            return "medium"
        elif score == 3:
            return "high"
        else:
            raise ValueError(f"Invalid score: {score}")

    def prepare_summary(self, documents: List[Document]) -> Tuple[TreeSummaryPrompt, int]:
        context_documents = []
        total_tokens = 0
        total_importance = 0
        for document in documents:
            if document is None:
                logging.info("FlexibleDocumentTreeSummarizer: prepare_summary document is None")
                continue
            document_tokens = len(self._encoding.encode(document.page_content))
            if total_tokens + document_tokens > self._input_limit:
                continue
            context_documents.append(document)
            total_tokens += document_tokens
            total_importance += self.importance_to_score(document.metadata["importance"])

        average_importance = self.score_to_importance(total_importance // len(context_documents))
        return TreeSummaryPrompt(documents=context_documents), average_importance

    async def _get_single_summary(self,  summary_prompt: TreeSummaryPrompt, average_importance: str) -> Document:
        try:
            nowStamp = datetime.now().timestamp()
            messages = [[summary_prompt.to_system_message(), 
                        summary_prompt.to_human_message()]]

            response = await self._llm.agenerate(messages)
            if not response.generations or not response.generations[0]:
                raise Exception("LLM did not provide a valid summary response.")

            summarized_content = response.generations[0][0].text
            metadata = {
                "id": random.randint(0, 2**32 - 1),
                "extra_index": summary_prompt.documents[0].metadata["extra_index"],
                "created_at": nowStamp,
                "importance": average_importance,
                "last_accessed_at": nowStamp,
                "summarizations": 0,
                "group_id": summary_prompt.documents[0].metadata["group_id"],
                "memory_type": MemoryType.SUBCONSCIOUS_MEMORY.value,
            }
            return Document(page_content=summarized_content,metadata=metadata)
        except Exception as e:
            if self._verbose:
                logging.warn(f"FlexibleDocumentSummarizer: _get_single_summary exception e: {e}\n{traceback.format_exc()}")
        return None

    # handle a single layer of the tree in parallel
    async def _get_layer_summaries(self, documents: List[Document]) -> List[Document]:
        tasks = []
        i = 0
        while i < len(documents):
            # Batch the documents using the token count limit.
            summary_prompt, average_importance = self.prepare_summary(documents[i:])
            batch_size = len(summary_prompt.documents)

            # Create an asynchronous task for summarizing the batched documents.
            if batch_size > 0:  # Check if there are documents to summarize
                task = self._get_single_summary(summary_prompt, average_importance)
                tasks.append(task)
                i += batch_size  # Move the index to the next unprocessed document.
            else:
                i += 1  # If a document is too large, skip it

        # Wait for all tasks to complete in parallel.
        return [doc for doc in await asyncio.gather(*tasks) if doc is not None]

    async def tree_summarize(self, documents: List[Document], new_docs: List[Document] = None) -> List[Document]:
        if not documents:
            return []
        if len(documents) == 1:
            return documents
        if new_docs is None:
            new_docs = []
        layer_summaries = await self._get_layer_summaries(documents)
        if not layer_summaries:
            return documents  # If no summaries are generated, return the original documents
        new_docs.extend(layer_summaries)
        return await self.tree_summarize(layer_summaries, new_docs)

    async def recursive_summary(self, documents: List[Document]) -> Document:
        new_docs = await self.tree_summarize(documents)
        if len(new_docs) > 1:
            return await self.recursive_summary(new_docs)
        else:
            return new_docs[0]

    async def aupdate_documents(self, documents: List[Document]):
        return await self.recursive_summary(documents)