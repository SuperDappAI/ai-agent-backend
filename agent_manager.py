import time
import shutil
from dotenv import load_dotenv
from reader_writer_lock import ReaderWriterLock
from pathlib import Path
import os
import json
import dill
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
from langchain.schema import Document
from copy import deepcopy

class MyTimeWeightedVectorStoreRetriever(TimeWeightedVectorStoreRetriever):
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        # SYSCOIN need to pop this because vectorstore does not expect this argument in kwargs
        current_time = kwargs.pop("current_time", datetime.now())
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)

class AgentManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./storage_memory"
        self.embeddings = OpenAIEmbeddings()
        self.memory = {}
        self.locks = {} 
        self.LLM = OpenAI()
        # Save function scheduled to run every 30 to 60 seconds
        schedule.every(30).to(60).seconds.do(self.save)
        
        # Create new thread for schedule
        self.stop_event = threading.Event()
        self.scheduler_thread = threading.Thread(target=self.run_continuously)
        self.scheduler_thread.start()

    def run_continuously(self):
        """Keep checking and running pending tasks every second."""
        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

    def stop(self):
        """Stops the scheduler thread."""
        self.save()
        self.stop_event.set()
        self.scheduler_thread.join()

    def get_user_lock(self, user_id):
        return self.locks.setdefault(user_id, ReaderWriterLock())

    def create_new_memory_retriever(self, path):
        """Create a new vector store retriever unique to the agent."""
        collection_name = "base_memory"
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
        except:
            print("AgentManager: couldn't create collection? It probably already exists, and loaded from disk...")
        finally:
            print(f"AgentManager: Creating memory store with collection {collection_name}")
            vectorstore = Qdrant(client, collection_name, self.embeddings)
            return MyTimeWeightedVectorStoreRetriever(
                vectorstore=vectorstore, decay_rate=0.001, other_score_keys=["importance"], k=15
            )

    def create_memory(self, path):
        return GenerativeAgentMemory(
            llm=self.LLM,
            memory_retriever=self.create_new_memory_retriever(path),
            verbose=False
        )

    def load(self, user_id):
        """Load existing index data from the filesystem for a specific user."""
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            start = time.time()
            userpath = Path(f"{self.dirpath}/{user_id}")
            mempath = Path(f"{self.dirpath}/{user_id}/memory_stream.wb")
            self.memory[user_id] = self.create_memory(userpath)
            if mempath.exists():
                print("AgentManager: Loading memory stream from disk")
                self.memory[user_id].memory_retriever.memory_stream = dill.load(open(mempath, "rb"))
            end = time.time()
            print(f"AgentManager: Load operation took {end - start} seconds")
        finally:
            lock.writer_release()

    def save(self):
        """Persist current index data to the filesystem."""
        for user_id, idx in self.memory.items():
            lock = self.get_user_lock(user_id)
            lock.writer_acquire()
            try:
                start = time.time()
                if lock.dirty:
                    userpath = Path(f"{self.dirpath}/{user_id}/memory_stream.wb")
                    with open(userpath, "wb") as f:
                        dill.dump(idx.memory_retriever.memory_stream, f)
                        lock.dirty = False
                        end = time.time()
                        print(f"AgentManager: Save operation for user {user_id} took {end - start} seconds")
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
            obj = {"user": query, "AiDA": llm_response}
            self.memory[user_id].save_context(
                {},
                {
                    self.memory[user_id].add_memory_key: json.dumps(obj),
                    self.memory[user_id].now_key: datetime.now(),
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

    def pull_memory(self, user_id, query):
        """Fetch memory based on a query for a specific user."""
        start = time.time()
        if user_id not in self.memory:
            self.load(user_id)
        lock = self.get_user_lock(user_id)
        lock.reader_acquire()
        response = None
        try:
            if user_id in self.memory:
                response = self.memory[user_id].fetch_memories(query, now=datetime.now())
        except Exception as e:
            print(f"AgentManager: pull_memory exception {e}")
        finally:
            lock.reader_release()
            end = time.time()
            print(f"AgentManager: pull_memory operation took {end - start} seconds")
            return response, end - start

    def get_latest_memories(self, user_id, token_count):
        """Fetch latest memories up to token_acount for a specific user."""
        token_count = 1200 if token_count is None else token_count
        start = time.time()
        if user_id not in self.memory:
            self.load(user_id)
        lock = self.get_user_lock(user_id)
        lock.reader_acquire()
        response = None
        try:
            if user_id in self.memory:
                old_limit = self.memory[user_id].max_tokens_limit
                self.memory[user_id].max_tokens_limit = token_count
                response = self.memory[user_id].load_memory_variables({self.memory[user_id].most_recent_memories_token_key: 0})
                self.memory[user_id].max_tokens_limit = old_limit
        except Exception as e:
            print(f"AgentManager: get_latest_memories exception {e}")
        finally:
            lock.reader_release()
            end = time.time()
            print(f"AgentManager: get_latest_memories token_count: {token_count} operation took {end - start} seconds")
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