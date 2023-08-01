import time
import shutil
from dotenv import load_dotenv
from llama_index import ServiceContext, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
# from llama_index.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index.indices.postprocessor import LLMRerank
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers import TreeSummarize
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
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./storage_web"
        self.retriever = {}
        self.locks = {}  # Dictionary to store locks for each hash
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, service_context=ServiceContext.from_defaults(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        ))
        self.summarizer = TreeSummarize(verbose=True, service_context=ServiceContext.from_defaults(
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

    async def get_retrieved_nodes(self, retriever, query_str: str):
        query_bundle = QueryBundle(query_str)
        retrieved_nodes = retriever.retrieve(query_bundle)
        # rerank and summarize if we need to select only up to top_n results
        if len(retrieved_nodes) > self.reranker._top_n:
            retrieved_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle)
            text_list = []
            for node_with_score in retrieved_nodes:
                # Extract the text from the TextNode
                text_list.append(node_with_score.node.text)
            retrieved_nodes = await self.summarizer.aget_response(query_str, text_list)
        return retrieved_nodes
    
    def load(self, hash_key):
        """Load existing index data from the filesystem for a specific hash."""
        start = time.time()
        lock = self.get_hash_lock(hash_key)
        lock.writer_acquire()
        try:
            hashpath = Path(f"{self.dirpath}/{hash_key}")
            if hashpath.exists() and hashpath.is_dir():
                print("WebManager: Loading from disk")
                storage_context = StorageContext.from_defaults(persist_dir=hashpath)
                self.retriever[hash_key] = VectorIndexRetriever(
                    index=load_index_from_storage(storage_context),
                    similarity_top_k=10
                )
        finally:
            lock.writer_release()
            end = time.time()
            print(f"WebManager: Load operation took {end - start} seconds")

    def save(self):
        """Persist current index data to the filesystem."""
        start = time.time()
        for hash_key, idx in self.retriever.items():
            lock = self.get_hash_lock(hash_key)
            lock.writer_acquire()
            try:
                if lock.dirty:
                    filepath = f"{self.dirpath}/{hash_key}"
                    idx._index.storage_context.persist(persist_dir=filepath)
                    lock.dirty = False
            finally:
                lock.writer_release()
        end = time.time()
        print(f"WebManager: Save operation took {end - start} seconds")

    async def search_html(self, function_input: HTMLInput):
        """Fetch HTML data based on a query for a specific hash."""
        start = time.time()
        if function_input.hash not in self.retriever:
            self.load(function_input.hash)
        lock = self.get_hash_lock(function_input.hash)
        lock.reader_acquire()
        response = None
        try:
            if function_input.hash not in self.retriever:
                documents = []
                for item in function_input.action_items:
                    text_splitter = SentenceSplitter()
                    chunks = text_splitter.split_text(text=item.html_doc)
                    documents.extend([Document(text=chunk, metadata={'url': item.source_url}) for chunk in chunks])
                lock.dirty = True
                self.retriever[function_input.hash] = VectorIndexRetriever(
                    index=VectorStoreIndex.from_documents(documents),
                    similarity_top_k=10
                )
            response = await self.get_retrieved_nodes(self.retriever[function_input.hash], function_input.query)
        except Exception as e:
            print(f"WebManager: search_html exception {e}")
        finally:
            lock.reader_release()
            end = time.time()
            print(f"WebManager: search_html operation took {end - start} seconds")
            return response, {end - start}

    def delete_html(self, hash_key):
        """Delete all memories for a specific hash."""
        start = time.time()
        lock = self.get_hash_lock(hash_key)
        lock.writer_acquire()
        try:
            hashpath = Path(f"{self.dirpath}/{hash_key}")
            if hashpath.exists() and hashpath.is_dir():
                shutil.rmtree(hashpath)
            self.retriever.pop(hash_key, None)
        finally:
            lock.writer_release()
            end = time.time()
            return {end - start}