import traceback
import time
import logging
import os
import itertools
import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from document_summarizer import FlexibleDocumentSummarizer
from document_tree_summarizer import FlexibleDocumentTreeSummarizer
from langchain.chat_models import ChatOpenAI
from apscheduler.triggers.date import DateTrigger

class MemorySummarizer:
    def __init__(self, agent_manager):
        os.getenv("OPENAI_API_KEY")
        self.agent_manager = agent_manager
        self.flexible_document_summarizer = FlexibleDocumentSummarizer(
            ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0), verbose=self.agent_manager.verbose
        )
        self.flexible_document_tree_summarizer = FlexibleDocumentTreeSummarizer(
            ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0), verbose=self.agent_manager.verbose
        )
        self.scheduler = AsyncIOScheduler(timezone="America/Los_Angeles")
        #self.scheduler.add_job(self.summarize_and_update_documents, 'cron', hour=23)
        self.scheduler.add_job(self.summarize_and_update_documents, 'interval', seconds=10)
        #self.scheduler.add_job(self.tree_summarize_and_update_documents, 'cron', hour=3)
        self.scheduler.add_job(self.tree_summarize_and_update_documents, 'interval', seconds=30)
        self.summarizing = False

    async def summarize_and_update_documents(self):
        if self.summarizing:
            logging.info("tree_summarize_and_update_documents: already summarizing")
            return
        self.summarizing = True
        while not self.agent_manager.stop_event.is_set():
            try:
                start = time.time()
                # Get the documents to summarize
                documents = self.agent_manager.memory.memory_retriever.base_retriever.get_documents_for_summarization()
                if len(documents) > 0:
                    await self.flexible_document_summarizer.aupdate_documents(documents)
                    # upsert entire document set to qdrant against existing IDs (stored in metadata)
                    ids = [doc.metadata["id"] for doc in documents]
                    await self.agent_manager.memory.memory_retriever.base_retriever.vectorstore.aadd_documents(documents, ids=ids)
                    end = time.time()
                    if self.flexible_document_summarizer._verbose:
                        logging.info(f"summarize_and_update_documents completed in {end-start}")
                else:
                    break          
            except Exception as e:
                logging.warn(f"MemorySummarizer: summarize_and_update_documents exception {e}\n{traceback.format_exc()}")
        self.summarizing = False
    
    async def tree_summarize_and_update_documents(self):
        if self.summarizing:
            logging.info("tree_summarize_and_update_documents: already summarizing")
            return
        self.summarizing = True
        while not self.agent_manager.stop_event.is_set():
            try:
                start = time.time()
                # Get the documents to summarize
                documents = self.agent_manager.memory.memory_retriever.base_retriever.get_documents_for_tree_summarization()

                # Stop if no documents were found
                if len(documents) > 0:
                    # Sort documents by user and conversation to ensure correct grouping
                    sorted_by_user_convo = sorted(documents, key=lambda x: (x.metadata["group_id"], x.metadata["extra_index"]))

                    # Group by user and conversation, then sort each group by time
                    groups = []
                    for _, group in itertools.groupby(sorted_by_user_convo, key=lambda x: (x.metadata["group_id"], x.metadata["extra_index"])):
                        sorted_group = sorted(list(group), key=lambda x: x.metadata["created_at"])
                        groups.append(sorted_group)

                    # Summarize each group asynchronously
                    summarization_tasks = [self.flexible_document_tree_summarizer.aupdate_documents(group) for group in groups]
                    all_new_docs = await asyncio.gather(*summarization_tasks)
                    # Flatten the list of new docs
                    # Check if every element in all_new_docs is a list
                    if all(isinstance(sublist, list) for sublist in all_new_docs):
                        flat_new_docs = [doc for sublist in all_new_docs for doc in sublist]
                    else:
                        flat_new_docs = all_new_docs

                    # Extract IDs and add new summary documents
                    ids = [doc.metadata["id"] for doc in flat_new_docs]
                    print(f"flat_new_docs adding {flat_new_docs}")
                    await self.agent_manager.memory.memory_retriever.base_retriever.vectorstore.aadd_documents(flat_new_docs, ids=ids)
                    # delete the summarized documents
                    self.agent_manager.memory.memory_retriever.base_retriever.delete_documents(documents)
                    end = time.time()
                    if self.flexible_document_summarizer._verbose:
                        logging.info(f"tree_summarize_and_update_documents completed in {end-start}")
                else:
                    break
            except Exception as e:
                logging.warn(f"MemorySummarizer: tree_summarize_and_update_documents exception {e}\n{traceback.format_exc()}")
        self.summarizing = False


    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.shutdown()
