from apscheduler.schedulers.asyncio import AsyncIOScheduler
from document_summarizer import FlexibleDocumentSummarizer
from langchain.chat_models import ChatOpenAI
import logging
import os

class MemorySummarizer:
    def __init__(self, agent_manager):
        os.getenv("OPENAI_API_KEY")
        self.agent_manager = agent_manager
        self.flexible_document_summarizer = FlexibleDocumentSummarizer(
            ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
        )
        self.scheduler = AsyncIOScheduler(timezone="America/Los_Angeles")
        self.scheduler.add_job(self.summarize_and_update_documents, 'cron', hour=1)

    async def summarize_and_update_documents(self):
        documents = None
        while not self.agent_manager.stop_event.is_set():
            try:
                # Get the documents to summarize
                documents = self.agent_manager.memory.memory_retriever.base_retriever.get_documents_for_summarization()
                if len(documents) <= 0:  # Stop if no documents were found
                    break
                await self.flexible_document_summarizer.aupdate_documents(documents)
                # upsert entire document set to qdrant against existing IDs (stored in metadata)
                ids = [doc.metadata["id"] for doc in documents]
                await self.agent_manager.memory.memory_retriever.base_retriever.vectorstore.aadd_documents(documents, ids=ids)
            except Exception as e:
                logging.warn(f"MemorySummarizer: exception {e}")
                break

    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.shutdown()
