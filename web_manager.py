import time
import shutil
from dotenv import load_dotenv
from llama_index import ServiceContext, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
# from llama_index.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index.indices.postprocessor import LLMRerank
from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.query.schema import QueryBundle
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
    query: str

class WebManager:
    scheduler = schedule.Scheduler()
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./storage_web"
        self.web_locks = {}
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, service_context=ServiceContext.from_defaults(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        ))
        self.scheduler.every(3600).seconds.do(self.prune_cache)

        # Create new thread for schedule
        self.stop_event = threading.Event()
        self.scheduler_thread = threading.Thread(target=self.run_continuously)
        self.scheduler_thread.start()

    def run_continuously(self):
        """Keep checking and running pending tasks every second."""
        while not self.stop_event.is_set():
            self.scheduler.run_pending()
            time.sleep(1)

    def get_lock(self, hash_key):
        # Return an existing lock for this user_id or create a new one
        return self.web_locks.setdefault(hash_key, ReaderWriterLock())

    def stop(self):
        """Stops the scheduler thread."""
        self.stop_event.set()
        self.scheduler_thread.join()

    def extract_text_and_source_url(self, retrieved_nodes):
        result = []
        for node_with_score in retrieved_nodes:
            text = node_with_score.node.text
            source_url = node_with_score.node.metadata.get('source_url')
            result.append({'text': text, 'source_url': source_url})
        return result

    async def get_retrieved_nodes(self, retriever, query_str: str):
        query_bundle = QueryBundle(query_str)
        retrieved_nodes = retriever.retrieve(query_bundle)
        retrieved_nodes[:] = [node for node in retrieved_nodes if node.score >= 0.6]
        # rerank if we need to select only up to top_n results
        if len(retrieved_nodes) > self.reranker._top_n:
            print(f"WebManager: Reranking {len(retrieved_nodes)} results down to {self.reranker._top_n}")
            retrieved_nodes[:] = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle)
        return retrieved_nodes
    
    def load(self, hash_key):
        """Load existing index data from the filesystem for a specific hash."""
        start = time.time()
        hashpath = Path(f"{self.dirpath}/{hash_key}")
        retriever = None
        if hashpath.exists() and hashpath.is_dir():
            print("WebManager: Loading from disk")
            storage_context = StorageContext.from_defaults(persist_dir=hashpath)
            retriever = VectorIndexRetriever(
                index=load_index_from_storage(storage_context),
                similarity_top_k=10
            )
        end = time.time()
        print(f"WebManager: Load operation took {end - start} seconds")
        return retriever

    def save(self, retriever, hash_key):
        """Persist current index data to the filesystem."""
        start = time.time()
        filepath = Path(f"{self.dirpath}/{hash_key}")
        if not filepath.exists() or not filepath.is_dir():
            retriever._index.storage_context.persist(persist_dir=filepath)
        end = time.time()
        print(f"WebManager: Save operation took {end - start} seconds")

    def delete_html(self, hash_key):
        """Delete all memories for a specific hash."""
        start = time.time()
        hashpath = Path(f"{self.dirpath}/{hash_key}")
        if hashpath.exists() and hashpath.is_dir():
            shutil.rmtree(hashpath)
        end = time.time()
        print(f"WebManager: Save operation took {end - start} seconds")
        return end - start
        
    async def search_html(self, function_input: HTMLInput):
        """Fetch HTML data based on a query for a specific hash."""
        start = time.time()
        web_lock = self.get_lock(function_input.hash)
        web_lock.writer_acquire()
        retriever = self.load(function_input.hash)
        response = None
        try:
            documents = []
            if retriever is None:
                for item in function_input.action_items:
                    text_splitter = SentenceSplitter()
                    chunks = text_splitter.split_text(text=item.html_doc)
                    documents.extend([Document(text=chunk, metadata={'source_url': item.source_url}) for chunk in chunks])
                retriever = VectorIndexRetriever(
                    index=VectorStoreIndex.from_documents(documents),
                    similarity_top_k=10
                )
                end = time.time()
                print(f"WebManager: Loaded from documents operation took {end - start} seconds")
            nodes = await self.get_retrieved_nodes(retriever, function_input.query)
            response = self.extract_text_and_source_url(nodes)
            self.save(retriever, function_input.hash)
        except Exception as e:
            print(f"WebManager: search_html exception {e}")
        finally:
            web_lock.writer_release()
            end = time.time()
            print(f"WebManager: search_html operation took {end - start} seconds")
            return response, end - start

    def delete_html(self, hash_key):
        """Delete all memories for a specific hash."""
        start = time.time()
        web_lock = self.get_lock(hash_key)
        web_lock.writer_acquire()
        try:
            hashpath = Path(f"{self.dirpath}/{hash_key}")
            if hashpath.exists() and hashpath.is_dir():
                shutil.rmtree(hashpath)
        finally:
            web_lock.writer_release()
            end = time.time()
            return end - start
            
    def prune_cache(self):
        """Prune cache that are older than an hour."""
        current_time = time.time()
        path = Path(self.dirpath)
        if path.exists() and path.is_dir():
            # Iterating through all directories in the dirpath
            for directory in os.scandir(self.dirpath):
                if directory.is_dir():
                    # Get the directory's last modified time
                    dir_time = directory.stat().st_mtime
                    # Check if the directory is older than an hour
                    if current_time - dir_time > 3600:
                        self.delete_html(directory.name)
            end = time.time()
            print(f"WebManager: prune_cache operation took {end - current_time} seconds")

    def does_hash_exist(self, hash_key):
        """Does the hash of the web content exist in our cache?."""
        start = time.time()
        hashpath = Path(f"{self.dirpath}/{hash_key}")
        exists = hashpath.exists() and hashpath.is_dir()
        end = time.time()
        return exists, end - start