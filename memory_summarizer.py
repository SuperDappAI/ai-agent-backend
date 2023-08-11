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
from langchain.schema import Document

class MemorySummarizer:
    def __init__(self, agent_manager):
        os.getenv("OPENAI_API_KEY")
        self.agent_manager = agent_manager
        self.flexible_document_summarizer = FlexibleDocumentSummarizer(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0), verbose=self.agent_manager.verbose
        )
        self.flexible_document_tree_summarizer = FlexibleDocumentTreeSummarizer(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0), verbose=self.agent_manager.verbose
        )
        self.scheduler = AsyncIOScheduler(timezone="America/Los_Angeles")
        self.scheduler.add_job(self.summarize_and_update_documents, 'cron', hour=23)
        #self.scheduler.add_job(self.summarize_and_update_documents, 'interval', seconds=30)
        self.scheduler.add_job(self.tree_summarize_and_update_documents, 'cron', hour=3)
        #self.scheduler.add_job(self.tree_summarize_and_update_documents, 'interval', seconds=90)
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
                        logging.info(f"summarize_and_update_documents completed in {end-start} seconds")
                else:
                    break          
            except Exception as e:
                logging.warn(f"MemorySummarizer: summarize_and_update_documents exception {e}\n{traceback.format_exc()}")
                break
        self.summarizing = False
    
    def _sort_and_group_documents(self, documents):
        """Sort and group documents by user and conversation."""
        # Sort by user and conversation
        sorted_by_user_convo = sorted(documents, key=lambda x: (x.metadata["group_id"], x.metadata["extra_index"]))
        # Group by user and conversation, then sort each group by time
        return [sorted(list(group), key=lambda x: x.metadata["created_at"]) 
                for _, group in itertools.groupby(sorted_by_user_convo, 
                                                  key=lambda x: (x.metadata["group_id"], x.metadata["extra_index"]))]

    async def _process_group(self, group):
        """Process a group of documents."""
        if len(group) == 1:
            group[0].metadata["summarizations"] = 100
            return [group[0]]

        return await self.flexible_document_tree_summarizer.aupdate_documents(group)

    async def tree_summarize_and_update_documents(self):
        """Tree summarize and update documents."""
        if self.summarizing:
            logging.info("Already summarizing")
            return

        self.summarizing = True
        while not self.agent_manager.stop_event.is_set():
            try:
                start = time.time()
                documents = self.agent_manager.memory.memory_retriever.base_retriever.get_documents_for_tree_summarization()
                if documents:
                    if self.flexible_document_tree_summarizer._verbose:
                        logging.info("Starting document processing...")
                    groups = self._sort_and_group_documents(documents)
                    all_new_docs = await asyncio.gather(*[self._process_group(group) for group in groups])
                    # Flatten the list and filter out empty lists
                    all_new_docs = [doc for sublist in all_new_docs for doc in sublist]
                    if all_new_docs:
                        if self.flexible_document_tree_summarizer._verbose:
                            logging.info("Adding new summarized documents and deleting originals...")
                        ids = [doc.metadata["id"] for doc in all_new_docs]
                        self.agent_manager.memory.memory_retriever.base_retriever.delete_documents(documents)
                        await self.agent_manager.memory.memory_retriever.base_retriever.vectorstore.aadd_documents(all_new_docs, ids=ids)
                    if self.flexible_document_tree_summarizer._verbose:
                        logging.info(f"Completed in {time.time()-start} seconds")
                else:
                    if self.flexible_document_tree_summarizer._verbose:
                        logging.info("No documents found for processing...")
                    break
            except Exception as e:
                logging.warn(f"Exception {e}\n{traceback.format_exc()}")
                break
        self.summarizing = False


    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.shutdown()
