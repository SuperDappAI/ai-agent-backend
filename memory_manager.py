import time
import shutil
from dotenv import load_dotenv
from llama_index import ServiceContext, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from llama_index.indices.postprocessor import LLMRerank
from reader_writer_lock import ReaderWriterLock
from pathlib import Path
import schedule
import threading
import os
import json

class MemoryManager1:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./storage_memory"
        self.index = {}
        self.query_engine = {}
        self.locks = {} 
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, 
            service_context=ServiceContext.from_defaults(
                llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
            ))

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
        self.stop_event.set()
        self.scheduler_thread.join()

    def get_user_lock(self, user_id):
        return self.locks.setdefault(user_id, ReaderWriterLock())

    def load(self, user_id):
        """Load existing index data from the filesystem for a specific user."""
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            print("MemoryManager: Loading from disk")
            start = time.time()
            userpath = Path(f"{self.dirpath}_{user_id}")
            if userpath.exists() and userpath.is_dir():
                storage_context = StorageContext.from_defaults(persist_dir=userpath)
                self.index[user_id] = load_index_from_storage(storage_context)
                self.query_engine[user_id] = self.index[user_id].as_query_engine(
                    similarity_top_k=10,
                    node_postprocessors=[self.reranker],
                    response_mode="tree_summarize"
                )
            end = time.time()
            print(f"MemoryManager: Load operation took {end - start} seconds")
        finally:
            lock.writer_release()

    def save(self):
        """Persist current index data to the filesystem."""
        for user_id, idx in self.index.items():
            lock = self.get_user_lock(user_id)
            lock.writer_acquire()
            try:
                start = time.time()
                if idx.dirty:
                    filepath = f"{self.dirpath}_{user_id}"
                    idx.storage_context.persist(persist_dir=filepath)
                    idx.dirty = False
                end = time.time()
                print(f"MemoryManager: Save operation for user {user_id} took {end - start} seconds")
            finally:
                lock.writer_release()

    def push_memory(self, user_id, query, llm_response):
        """Add new memory to the current index for a specific user."""
        if user_id not in self.index:
            self.load(user_id)
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            start = time.time()
            # Create a dictionary to represent the memory data
            memory = {
                "user": query,
                "assistant": llm_response
            }

            # Convert the dictionary to a JSON string using json.dumps
            obj = json.dumps(memory)
            doc = Document(text=obj)
            if user_id not in self.index:
                self.index[user_id] = VectorStoreIndex.from_documents([doc])
            else:
                self.index[user_id].update(doc)
            self.index[user_id].dirty = True
            self.query_engine[user_id] = self.index[user_id].as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[self.reranker],
                response_mode="tree_summarize"
            )
            end = time.time()
            print(f"MemoryManager: push_memory operation took {end - start} seconds")
        finally:
            lock.writer_release()

    def pull_memory(self, user_id, query):
        """Fetch memory based on a query for a specific user."""
        lock = self.get_user_lock(user_id)
        lock.reader_acquire()
        try:
            if user_id not in self.query_engine:
                print("MemoryManager: Error pull_memory, hash doesn't exist")
                return None
            start = time.time()
            response = self.query_engine[user_id].query(query)
            end = time.time()
            print(f"MemoryManager: pull_memory operation took {end - start} seconds")
            return response
        finally:
            lock.reader_release()

    def delete_memory(self, user_id):
        """Delete all memories for a specific user."""
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            userpath = Path(f"{self.dirpath}_{user_id}")
            if userpath.exists() and userpath.is_dir():
                shutil.rmtree(userpath)
            self.index.pop(user_id, None)
            self.query_engine.pop(user_id, None)
        finally:
            lock.writer_release()