import time
import shutil
from dotenv import load_dotenv
from llama_index import ServiceContext, Document, VectorStoreIndex, StorageContext, load_index_from_storage
# from llama_index.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from llmreranker import LLMRerank
from reader_writer_lock import ReaderWriterLock
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
import schedule
import threading
import os

class HTMLItem(BaseModel):
    source_url: str
    html_doc: str

class HTMLInput(BaseModel):
    action_items: List[HTMLItem] = Field(..., example=[{"source_url": "http://example.com", "html_doc": "text1"}])
    hash: str

class WebManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./storage_web"
        self.index = {}
        self.query_engine = {}
        self.locks = {}  # Dictionary to store locks for each hash
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, service_context=ServiceContext.from_defaults(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        ))
        schedule.every(300).to(600).seconds.do(self.save)

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

    def get_hash_lock(self, hash_key):
        return self.locks.setdefault(hash_key, ReaderWriterLock())


    def load(self, hash_key):
        """Load existing index data from the filesystem for a specific hash."""
        start = time.time()
        lock = self.get_hash_lock(hash_key)
        lock.writer_acquire()
        try:
            print("WebManager: Loading from disk")
            hashpath = Path(f"{self.dirpath}_{hash_key}")
            if hashpath.exists() and hashpath.is_dir():
                storage_context = StorageContext.from_defaults(persist_dir=hashpath)
                self.index[hash_key] = load_index_from_storage(storage_context)
                self.query_engine[hash_key] = self.index[hash_key].as_query_engine(
                    similarity_top_k=10,
                    node_postprocessors=[self.reranker],
                    response_mode="tree_summarize"
                )
        finally:
            lock.writer_release()
            end = time.time()
            print(f"WebManager: Load operation took {end - start} seconds")

    def save(self):
        """Persist current index data to the filesystem."""
        start = time.time()
        for hash_key, idx in self.index.items():
            lock = self.get_hash_lock(hash_key)
            lock.writer_acquire()
            try:
                if lock.dirty:
                    filepath = f"{self.dirpath}_{hash_key}"
                    idx.storage_context.persist(persist_dir=filepath)
                    lock.dirty = False
            finally:
                lock.writer_release()
        end = time.time()
        print(f"WebManager: Save operation took {end - start} seconds")

    def push_html(self, function_input: HTMLInput):
        """Add new HTML data to the current index for a specific hash."""
        start = time.time()
        hash_key = function_input.hash
        if hash_key in self.index:
            print("WebManager: Error push_html, hash already exists")
            end = time.time()
            return {end - start}
        lock = self.get_hash_lock(hash_key)
        lock.writer_acquire()
        try:
            documents = [Document(text=item.html_doc, metadata={'url': item.source_url}) for item in function_input.action_items]
            self.index[hash_key] = VectorStoreIndex.from_documents(documents)
            lock.dirty = True
            self.query_engine[hash_key] = self.index[hash_key].as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[self.reranker],
                response_mode="tree_summarize"
            )
        finally:
            lock.writer_release()
            end = time.time()
            print(f"WebManager: push_html operation took {end - start} seconds")
            return {end - start}

    def pull_html(self, hash_key, query):
        """Fetch HTML data based on a query for a specific hash."""
        start = time.time()
        lock = self.get_hash_lock(hash_key)
        lock.reader_acquire()
        response = None
        try:
            if hash_key in self.query_engine:
                try:
                    self.reranker.query_str = query
                    response = self.query_engine[hash_key].query(query)
                except Exception as e:
                    print(f"WebManager: pull_html exception {e}")
        finally:
            lock.reader_release()
            end = time.time()
            print(f"WebManager: pull_html operation took {end - start} seconds")
            return response, {end - start}

    def delete_html(self, hash_key):
        """Delete all memories for a specific hash."""
        start = time.time()
        lock = self.get_hash_lock(hash_key)
        lock.writer_acquire()
        try:
            hashpath = Path(f"{self.dirpath}_{hash_key}")
            if hashpath.exists() and hashpath.is_dir():
                shutil.rmtree(hashpath)
            self.index.pop(hash_key, None)
            self.query_engine.pop(hash_key, None)
        finally:
            lock.writer_release()
            end = time.time()
            return {end - start}