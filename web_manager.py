import time
import shutil
from dotenv import load_dotenv
from llama_index import ServiceContext, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from llama_index.indices.postprocessor import LLMRerank
from pathlib import Path
import schedule
import threading
import os

class WebManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./web"
        self.index = {}
        self.query_engine = {}
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, service_context=ServiceContext.from_defaults(
            llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
        ))
        schedule.every(300).to(600).seconds.do(self.save)
        
        # Create new thread for schedule
        with threading.Thread(target=self.run_continuously) as thread:
            thread.start()

    def run_continuously(self):
        """Keep checking and running pending tasks every second."""
        while True:
            schedule.run_pending()
            time.sleep(1)

    def load(self, hash):
        """Load existing index data from the filesystem for a specific hash."""
        print("WebManager: Loading from disk")
        start = time.time()
        hashpath = Path(f"{self.dirpath}_{hash}")
        if hashpath.exists() and hashpath.is_dir():
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=hashpath)

            # load index
            self.index[hash] = load_index_from_storage(storage_context)
            self.query_engine[hash] = self.index[hash].as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[self.reranker],
                response_mode="tree_summarize"
            )
        end = time.time()
        print(f"WebManager: Load operation took {end - start} seconds")

    def save(self):
        """Persist current index data to the filesystem."""
        start = time.time()
        self.saving = True
        for hash_key in self.index:
            if self.index[hash_key].dirty is True:
                self.index[hash_key].storage_context.persist(persist_dir=f"{self.dirpath}_{hash_key}")
                self.index[hash_key].dirty = False
        self.saving = False
        end = time.time()
        print(f"WebManager: Save operation took {end - start} seconds")

    def push_html(self, hash, urls, html_docs):
        """Add new HTML data to the current index for a specific hash."""
        if hash in self.index:
            print("WebManager: Error push_html, hash already exists")
            return
        start = time.time()
        documents = [Document(t) for t in html_docs]
        self.index[hash] = VectorStoreIndex.from_documents(documents)
        for idx, doc in enumerate(documents):
            doc.extra_info.url = urls[idx]
        self.index[hash].dirty = True
        self.query_engine[hash] = self.index[hash].as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[self.reranker],
            response_mode="tree_summarize"
        )
        end = time.time()
        print(f"WebManager: push_html operation took {end - start} seconds")

    def pull_html(self, hash, query):
        """Fetch HTML data based on a query for a specific hash."""
        if hash not in self.query_engine:
            print("WebManager: Error pull_html, hash doesn't exists")
            return None
        start = time.time()
        response = self.query_engine[hash].query(
            query, 
        )
        end = time.time()
        print(f"WebManager: pull_html operation took {end - start} seconds")
        return response

    def delete_memory(self, hash):
        """Delete all memories for a specific hash."""
        hashpath = Path(f"{self.dirpath}_{hash}")
        if hashpath.exists() and hashpath.is_dir():
            shutil.rmtree(hashpath)
        self.index.pop(hash, None)
        self.query_engine.pop(hash, None)