import time
import shutil
from dotenv import load_dotenv
from reader_writer_lock import ReaderWriterLock
from pathlib import Path
import os
import schedule
import threading
from datetime import datetime
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_experimental.generative_agents import GenerativeAgentMemory
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from typing import Any, List
from qdrant_client.http.models import PayloadSchemaType

class AgentManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./storage_memory"
        self.embeddings = OpenAIEmbeddings()
        self.memory = {}
        self.locks = {} 
        self.payload_conversation_index_key = "metadata.conversation"
        self.LLM = OpenAI()

    def get_user_lock(self, user_id):
        return self.locks.setdefault(user_id, ReaderWriterLock())

    def create_new_memory_retriever(self, user_id, path):
        """Create a new vector store retriever unique to the agent."""
        collection_name = user_id
        client = QdrantClient(path=path)
        # create collection if it doesn't exist (if it exists it will fall into finally)
        try:
            client.create_collection(
                on_disk_payload=True,
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size = 1536,
                    distance = rest.Distance.COSINE,
                ),
            )
            client.create_payload_index(collection_name, self.payload_conversation_index_key, field_schema=PayloadSchemaType.KEYWORD)
        except:
            print("AgentManager: couldn't create collection? It probably already exists, and loaded from disk...")
        finally:
            print(f"AgentManager: Creating memory store with collection {collection_name}")
            vectorstore = Qdrant(client, collection_name, self.embeddings)
            return TimeWeightedVectorStoreRetriever(
                vectorstore=vectorstore, decay_rate=0.001, search_kwargs={"score_threshold":0.72}, other_score_keys=["importance"], k=15
            )

    def create_memory(self, user_id, path):
        return GenerativeAgentMemory(
            llm=self.LLM,
            memory_retriever=self.create_new_memory_retriever(user_id, path),
            verbose=False
        )

    def load(self, user_id):
        """Load existing index data from the filesystem for a specific user."""
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            start = time.time()
            userpath = Path(f"{self.dirpath}/{user_id}")
            self.memory[user_id] = self.create_memory(user_id, userpath)
            end = time.time()
            print(f"AgentManager: Load operation took {end - start} seconds")
        finally:
            lock.writer_release()



    def push_memory(self, user_id, query, llm_response):
        """Add new memory to the current index for a specific user."""
        start = time.time()
        if user_id not in self.memory:
            self.load(user_id)
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            self.memory[user_id].save_context(
                {
                    self.memory[user_id].add_user_key: query,
                    self.memory[user_id].add_aida_key: llm_response,
                    self.memory[user_id].now_key: datetime.now(),
                    self.memory[user_id].payload_conversation_key: user_id,
                },
            )
            lock.dirty = True
        except Exception as e:
            print(f"AgentManager: push_memory exception {e}") 
        finally:
            lock.writer_release()
            end = time.time()
            print(f"AgentManager: push_memory operation took {end - start} seconds")
            return end - start

    def pull_memory(self, user_id, convo_id, query):
        """Fetch memory based on a query for a specific user."""
        start = time.time()
        if user_id not in self.memory:
            self.load(user_id)
        lock = self.get_user_lock(user_id)
        lock.reader_acquire()
        response = None
        try:
            if user_id in self.memory:
                response = self.memory[user_id].load_memory_variables(
                {
                    self.memory[user_id].queries_key: [query],
                    self.memory[user_id].now_key: datetime.now(),
                    self.memory[user_id].payload_conversation_key: convo_id,
                }
            )
        except Exception as e:
            print(f"AgentManager: pull_memory exception {e}")
        finally:
            lock.reader_release()
            end = time.time()
            print(f"AgentManager: pull_memory operation took {end - start} seconds")
            return response, end - start

    def delete_memory(self, user_id):
        """Delete all memories for a specific user."""
        start = time.time()
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            userpath = Path(f"{self.dirpath}/{user_id}")
            if userpath.exists() and userpath.is_dir():
                shutil.rmtree(userpath)
            self.memory.pop(user_id, None)
        finally:
            lock.writer_release()
            end = time.time()
            return end - start