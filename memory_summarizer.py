import traceback
import time
import logging
import os

from datetime import datetime
from dotenv import load_dotenv
from typing import Any, Dict, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from document_summarizer import FlexibleDocumentSummarizer
from langchain.chat_models import ChatOpenAI
from apscheduler.triggers.date import DateTrigger
from langchain.schema import Document
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType
from langchain.retrievers import ContextualCompressionRetriever
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.embeddings import OpenAIEmbeddings
from generative_conversation_summarized_memory import GenerativeAgentConversationSummarizedMemory

class MemorySummarizer:
    def __init__(self, agent_manager):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")
        os.getenv("COHERE_API_KEY")
        os.getenv("OPENAI_API_KEY")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.embeddings = OpenAIEmbeddings()
        self.retriever = None
        self.agent_manager = agent_manager
        self.flexible_document_summarizer = FlexibleDocumentSummarizer(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0), verbose=self.agent_manager.verbose
        )
        self.scheduler = AsyncIOScheduler(timezone="America/Los_Angeles")
        self.scheduler.add_job(self.summarize_and_update_documents, 'cron', hour=23)
        #self.scheduler.add_job(self.summarize_and_update_documents, 'interval', seconds=30)
        self.summarizing = False
        self.load()

    def create_new_conversation_summarizer(self):
            """Create a new vector store retriever unique to the agent."""
            collection_name = "aida_conversation_summaries"
            client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
            # create collection if it doesn't exist (if it exists it will fall into finally)
            try:
                client.create_collection(
                    on_disk_payload=True,
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(
                        size=1536,
                        distance=rest.Distance.COSINE,
                    ),
                )
                # only used in reflection which isn't time critical so keep the index out for now unless reflection is very slow
                #client.create_payload_index(collection_name, self.payload_groupid_index_key, field_schema=PayloadSchemaType.KEYWORD)
                client.create_payload_index(collection_name, "metadata.extra_index", field_schema=PayloadSchemaType.KEYWORD)
                # ditto for summarizer
                #client.create_payload_index(collection_name, "metadata.importance", field_schema=PayloadSchemaType.INTEGER)
                #client.create_payload_index(collection_name, self.payload_lastaccessed_index_key, field_schema=PayloadSchemaType.FLOAT)
            except:
                print("MemorySummarizer: loaded from cloud...")
            finally:
                logging.info(f"MemorySummarizer: Creating memory store with collection {collection_name}")
                vectorstore = Qdrant(client, collection_name, self.embeddings)
                compressor = CohereRerank()
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                        collection_name=collection_name, client=client, vectorstore=vectorstore,
                    )
                )
                return compression_retriever
            
    def create_summarized_memory(self):
        return GenerativeAgentConversationSummarizedMemory(
            llm=self.agent_manager.LLM,
            memory_retriever=self.create_new_conversation_summarizer(),
            verbose=self.agent_manager.verbose
        )

    def load_memory_variables(self, **kwargs) -> Dict[str, str]:
        return self.retriever.load_memory_variables(**kwargs)

    def load(self):
        """Load existing index data from the cloud."""
        start = time.time()
        self.retriever = self.create_summarized_memory()
        end = time.time()
        logging.info(f"AgentManager: Load operation took {end - start} seconds")
       
    async def summarize_and_update_documents(self):
        if self.summarizing:
            logging.info("tree_summarize_and_update_documents: already summarizing")
            return
        self.summarizing = True
        while not self.agent_manager.stop_event.is_set():
            try:
                start = time.time()
                # Delete memories flagged as too old
                self.agent_manager.memory.memory_retriever.base_retriever.delete_max_summarized()
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
    

    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.shutdown()
