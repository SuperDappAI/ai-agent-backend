import time
import logging
from dotenv import load_dotenv
import os
import random
import threading
import asyncio
from datetime import datetime
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from qdrant_retriever import QDrantVectorStoreRetriever
from generative_memory import GenerativeAgentMemory
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType
from memory_summarizer import MemorySummarizer
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

class MemoryInput(BaseModel):
    user_id: str
    query: str
    conversation_id: str
    num_semantic_results: int = Field(..., example=10)
    similarity_threshold: float = Field(..., example=0.72)

class MemoryOutput(BaseModel):
    user_id: str
    query: str
    llm_response: str
    conversation_id: str
    importance: int

class ClearMemory(BaseModel):
    user_id: str
    conversation_id: str

class AgentManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.embeddings = OpenAIEmbeddings()
        self.LLM = OpenAI()
        self.memory = None
        self.load()
        # Create an instance of MemorySummarizer
        self.memory_summarizer = MemorySummarizer(agent_manager=self)
        self.memory_summarizer.start()
        self.thread_pool = ThreadPoolExecutor(max_workers=100)
        self.stop_event = threading.Event()

    def stop(self):
        self.memory_summarizer.stop()
        self.stop_event.set() # Signal to all threads to stop
        self.thread_pool.shutdown()

    async def save_and_reflect(self, id, memory_output: MemoryOutput):
        if len(id) > 0 and memory_output.importance >= 9:
            logging.info(f"memory importance {memory_output.importance}, queuing to reflect")
            asyncio.create_task(self.pause_to_reflect(id[0], memory_output))

    async def save_context_and_call_reflect(self, memory_output: MemoryOutput):
        id = await self.memory.save_context(memory_output.dict())
        asyncio.create_task(self.save_and_reflect(id, memory_output))

    async def pause_to_reflect(self, id_to_skip, memory_output: MemoryOutput):
        delay = random.randint(5, 300) # Delay in seconds, between 5 seconds and 5 minutes
        # Define a regular function that runs the coroutine
        def run_coroutine():
            asyncio.run(self._pause_to_reflect(id_to_skip, memory_output, delay))

        # Submit the regular function to the thread pool
        self.thread_pool.submit(run_coroutine)

    async def _pause_to_reflect(self, id_to_skip, memory_output: MemoryOutput, delay):
        start_time = time.time()
        while time.time() - start_time < delay:
            if self.stop_event.is_set():
                print("AgentManager: _pause_to_reflect was interrupted")
                return
            time.sleep(1) # Sleep for short periods and check again
        start = time.time()
        try:
            asyncio.create_task(self.memory.pause_to_reflect(id_to_skip, memory_output.user_id, memory_output.query, memory_output.conversation_id, now=datetime.now()))
        finally:
            end = time.time()
            logging.info(f"AgentManager: _pause_to_reflect operation took {end - start} seconds")
            
    async def push_memory(self, memory_output: MemoryOutput):
        """Add new memory to the current index for a specific user."""
        start = time.time()
        try:
            # This will start executing the function but not await its completion
            asyncio.create_task(self.save_context_and_call_reflect(memory_output))
        except Exception as e:
            logging.warn(f"AgentManager: push_memory exception {e}") 
        finally:
            end = time.time()
            logging.info(f"AgentManager: push_memory operation took {end - start} seconds")
            return end - start

    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        collection_name = "aida_memory"
        client = QdrantClient(location="https://5a136df6-42b6-4ff0-a1ac-a7a34101b901.eu-central-1-0.aws.cloud.qdrant.io:6333", port=6333, api_key=self.QDRANT_API_KEY)
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
            #client.create_payload_index(collection_name, self.payload_importance_index_key, field_schema=PayloadSchemaType.INTEGER)
            #client.create_payload_index(collection_name, self.payload_lastaccessed_index_key, field_schema=PayloadSchemaType.FLOAT)
        except:
            print("AgentManager: loaded from disk...")
        finally:
            logging.info(
                f"AgentManager: Creating memory store with collection {collection_name}")
            vectorstore = Qdrant(client, collection_name, self.embeddings)
            return QDrantVectorStoreRetriever(
                collection_name=collection_name, client=client, vectorstore=vectorstore
            )

    def create_memory(self):
        return GenerativeAgentMemory(
            llm=self.LLM,
            memory_retriever=self.create_new_memory_retriever(),
            verbose=True
        )

    def load(self):
        """Load existing index data from the filesystem for a specific user."""
        start = time.time()
        self.memory = self.create_memory()
        end = time.time()
        logging.info(
            f"AgentManager: Load operation took {end - start} seconds")

    def pull_memory(self, memory_input: MemoryInput):
        """Fetch memory based on a query for a specific user."""
        start = time.time()
        response = None
        try:
            response = self.memory.load_memory_variables(
                queries=[memory_input.query], 
                conversation_id=memory_input.conversation_id, 
                score_threshold=memory_input.similarity_threshold,
                k=memory_input.num_semantic_results,
            )
        except Exception as e:
            logging.info(f"AgentManager: pull_memory exception {e}")
        finally:
            end = time.time()
            logging.info(
                f"AgentManager: pull_memory operation took {end - start} seconds")
            return response, end - start

    def clear_conversation(self, clear_memory: ClearMemory):
        """Delete all memories for a specific conversation with a user."""
        start = time.time()
        try:
            self.memory.clear(clear_memory.conversation_id)
        except Exception as e:
            logging.info(f"AgentManager: clear_conversation exception {e}")
        finally:
            end = time.time()
            logging.info(
                f"AgentManager: clear_conversation operation took {end - start} seconds")
            return "success", end - start
    
