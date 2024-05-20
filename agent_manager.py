import time
import logging
import os
import asyncio
import traceback
import cachetools.func

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from qdrant_retriever import QDrantVectorStoreRetriever
from cohere_rerank import CohereRerank
from generative_memory import GenerativeAgentMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType
from memory_summarizer import MemorySummarizer
from pydantic import BaseModel
from document_summarizer import FlexibleDocumentSummarizer
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from datetime import datetime, timedelta
from typing import Any, Dict
from preferences_resolver import PreferencesResolver
from preferences_updater import PreferencesUpdater

class MemoryInput(BaseModel):
    api_key: str
    user_id: str
    query: str
    conversation_id: str
    summary: bool
    def __str__(self):
        return str(self.summary) + self.user_id + self.query + self.conversation_id

    def __eq__(self,other):
        return self.user_id == other.user_id and self.query == other.query and self.conversation_id == other.conversation_id and self.summary == other.summary

    def __hash__(self):
        return hash(str(self))

class MemoryOutput(BaseModel):
    api_key: str
    user_id: str
    query: str
    llm_response: str
    conversation_id: str

class ClearMemory(BaseModel):
    user_id: str
    conversation_id: str
    
class AgentManager:
    def __init__(self, rate_limiter, rate_limiter_sync):
        load_dotenv()  # Load environment variables
        os.getenv("COHERE_API_KEY")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.rate_limiter = rate_limiter
        self.rate_limiter_sync = rate_limiter_sync
        self.client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        self.verbose = True
        self.preferences_resolver = PreferencesResolver()
        self.preferences_updater = PreferencesUpdater(self.preferences_resolver, self.verbose)

    async def push_memory(self, memory_output: MemoryOutput):
        """Add new memory to the current index for a specific user."""
        start = time.time()
        memory = self.load(memory_output.api_key, memory_output.user_id)
        try:
            # update preferences on every exchange but only save summarized memory of a "finished" exchange, reflect on an important summarized memory and then decay memories
            asyncio.create_task(memory.pause_to_reflect(memory_output.dict(), self.preferences_resolver))
            asyncio.create_task(self.preferences_updater.update_preferences(ChatOpenAI(openai_api_key=memory_output.api_key, model="gpt-3.5-turbo-0125", temperature=0), memory_output.query, memory_output.llm_response, memory_output.user_id))
            # decay memory by summarizing it continiously until max_summarizations then prune
            asyncio.create_task(memory.decay())
        except Exception as e:
            logging.warn(f"AgentManager: push_memory exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(f"AgentManager: push_memory operation took {end - start} seconds")
            return end - start

    def create_new_memory_retriever(self, api_key: str, user_id: str):
        """Create a new vector store retriever unique to the agent."""
        collection_name = user_id
        # create collection if it doesn't exist (if it exists it will fall into finally)
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=1536,
                    distance=rest.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(collection_name, "metadata.extra_index", field_schema=PayloadSchemaType.KEYWORD)
        except:
            print("AgentManager: loaded from cloud...")
        finally:
            logging.info(f"AgentManager: Creating memory store with collection {collection_name}")
            vectorstore = Qdrant(self.client, collection_name, OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key))
            compressor = CohereRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                    rate_limiter=self.rate_limiter, rate_limiter_sync=self.rate_limiter_sync, collection_name=collection_name, client=self.client, vectorstore=vectorstore,
                )
            )
            return compression_retriever

    def create_memory(self, api_key: str, user_id: str):
        return GenerativeAgentMemory(
            rate_limiter=self.rate_limiter,
            llm=ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo-0125", max_tokens=1024),
            memory_retriever=self.create_new_memory_retriever(api_key, user_id),
            memory_summarizer=MemorySummarizer(rate_limiter=self.rate_limiter, rate_limiter_sync=self.rate_limiter_sync, flexible_document_summarizer=FlexibleDocumentSummarizer(ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo-0125", temperature=0), verbose=self.verbose), agent_manager=self),
            verbose=self.verbose
        )

    @cachetools.func.ttl_cache(maxsize=16384, ttl=36000)
    def load(self, api_key: str, user_id: str) -> GenerativeAgentMemory:
        """Load existing index data from the filesystem for a specific user."""
        start = time.time()
        memory = self.create_memory(api_key, user_id)
        end = time.time()
        logging.info(
            f"AgentManager: Load operation took {end - start} seconds")
        return memory

    def _document_from_scored_point(
        cls,
        scored_point: Any,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        return Document(
            page_content=scored_point.payload.get(content_payload_key),
            metadata=scored_point.payload.get(metadata_payload_key) or {},
        )

    def get_key_value_document(self, collection_name, key, value) -> Document:
        """Get the key value from vectordb via scrolling."""
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key=key, 
                    match=rest.MatchValue(value=value), 
                )
            ]
        )
        record, _ = self.client.scroll(collection_name=collection_name, scroll_filter=filter, limit = 1)
        if record is not None and len(record) > 0:
            return self._document_from_scored_point(
                record[0], "page_content", "metadata"
            )
        else:
            return None
    
    def load_summary(self, memory_input: MemoryInput) -> Dict[str, str]:
        doc = self.get_key_value_document(f"{memory_input.user_id}_summaries", "metadata.extra_index", memory_input.conversation_id)
        ret = ""
        if doc:
            ret = self.format_summary_simple(doc)
            return {
                "relevant_summary": ret,
            }
        return {}

    async def load_memory(self, memory_input: MemoryInput):
        memory = self.load(memory_input.api_key, memory_input.user_id)
        return await memory.load_memory_variables(
            queries=[memory_input.query], 
            conversation_id=memory_input.conversation_id
        )
    
    def _time_ago(self, timestamp: float) -> str:
        """Return a rough string representation of the time passed since a timestamp."""
        delta = datetime.now() - datetime.fromtimestamp(timestamp)
        if delta < timedelta(minutes=1):
            return "just now"
        elif delta < timedelta(hours=1):
            return f"{int(delta.total_seconds() / 60)} minutes ago"
        elif delta < timedelta(days=1):
            return f"{int(delta.total_seconds() / 3600)} hours ago"
        else:
            return f"{int(delta.total_seconds() / 86400)} days ago"

    def format_summary_simple(self, conversation_summary: Document) -> str:
        now = datetime.now().timestamp()
        created_at = conversation_summary.metadata.get("created_at", now)
        created_ago = self._time_ago(created_at)
        
        # Extracting the extra_index (conversation_id)
        conversation_id = conversation_summary.metadata.get("extra_index", "N/A")
        return f"(created: {created_ago}, conversation_id: {conversation_id}) {conversation_summary.page_content}"
  
    async def pull_memory(self, memory_input: MemoryInput):
        """Fetch memory based on a query for a specific user."""
        start = time.time()
        response = {}
        try:
            # look up from summary or semantically
            if memory_input.summary:
                if len(memory_input.conversation_id) <= 0:
                    logging.warn(f"AgentManager: pull_memory asked for summary but no conversation_id provided!")
                    end = time.time()
                    return {}, end - start
                response = self.load_summary(memory_input)
            else: 
                response = await self.load_memory(memory_input)
        except Exception as e:
            logging.warn(f"AgentManager: pull_memory exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(
                f"AgentManager: pull_memory operation took {end - start} seconds")
            return response, end - start

    def clear_collection_with_extra_index(self, collection_name, extra_index) -> None:
        """Clear memory contents."""
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.extra_index", 
                    match=rest.MatchValue(value=extra_index), 
                )
            ]
        )
        self.client.delete(collection_name=collection_name, points_selector=filter)

    def clear_conversation(self, clear_memory: ClearMemory):
        """Delete all memories for a specific conversation with a user."""
        start = time.time()
        try:
            self.clear_collection_with_extra_index(clear_memory.user_id, clear_memory.conversation_id)
            self.clear_collection_with_extra_index(f"{clear_memory.user_id}_summaries", clear_memory.conversation_id)
        except Exception as e:
            logging.warn(f"AgentManager: clear_conversation exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(
                f"AgentManager: clear_conversation operation took {end - start} seconds")
            return "success", end - start
    
