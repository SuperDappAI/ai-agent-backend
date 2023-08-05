from apscheduler.schedulers.asyncio import AsyncIOScheduler
from document_summarizer import FlexibleDocumentSummarizer
from langchain.llms import OpenAI
from llama_index.indices.service_context import ServiceContext
import asyncio
import logging

class MemorySummarizer:
    def __init__(self, agent_manager):
        self.agent_manager = agent_manager
        self.scheduler = AsyncIOScheduler(timezone="America/Los_Angeles")
        self.scheduler.add_job(self.summarize_and_update_documents, 'cron', hour=1)
        self.flexible_document_summarizer = FlexibleDocumentSummarizer(service_context=ServiceContext.from_defaults(
            llm=OpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        ))

    async def summarize_and_update_documents(self):
        documents = None
        while not self.agent_manager.stop_event.is_set():
            try:
                # Get the documents to summarize
                documents = self.agent_manager.memory.memory_retriever.get_documents_for_summarization()
                if not documents:  # Stop if no documents were found
                    break
                self.flexible_document_summarizer.update_documents(documents)
            except Exception as e:
                logging.warn(f"MemorySummarizer: exception {e}")
                break
        # upsert entire document set to qdrant against existing IDs (stored in metadata)
        if documents:
            ids = [doc.metadata["id"] for doc in documents]
            for doc in documents:
                doc.metadata.pop('relevance_score', None)
            asyncio.create_task(self.agent_manager.memory.memory_retriever.vectorstore.aadd_documents(documents, ids=ids, wait = False))

    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.shutdown()
