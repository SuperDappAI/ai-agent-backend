import time
import shutil
from dotenv import load_dotenv
from llama_index import OpenAI, ServiceContext, LLMRerank, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from pathlib import Path
import schedule
import threading
import os

class MemoryManager1:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./memory"
        self.index = {}
        self.query_engine = {}
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, service_context=ServiceContext.from_defaults(
            llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
        ))
        # Save function scheduled to run every 30 to 60 seconds
        schedule.every(30).to(60).seconds.do(self.save)
        
        # Create new thread for schedule
        with threading.Thread(target=self.run_continuously) as thread:
            thread.start()

    def run_continuously(self):
        """Keep checking and running pending tasks every second."""
        while True:
            schedule.run_pending()
            time.sleep(1)

    def load(self, user_id):
        """Load existing index data from the filesystem for a specific user."""
        print("MemoryManager: Loading from disk")
        start = time.time()
        userpath = Path(f"{self.dirpath}_{user_id}")
        if userpath.exists() and userpath.is_dir():
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=userpath)

            # load index
            self.index[user_id] = load_index_from_storage(storage_context)
            self.query_engine[user_id] = self.index[user_id].as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[self.reranker],
                response_mode="tree_summarize"
            )
        end = time.time()
        print(f"MemoryManager: Load operation took {end - start} seconds")

    def save(self):
        """Persist current index data to the filesystem."""
        start = time.time()
        self.saving = True
        for idx in self.index:
            if self.index[idx].dirty is True:
                filepath = f"{self.dirpath}_{idx}"
                self.index[idx].storage_context.persist(persist_dir=filepath)
                self.index[idx].dirty = False
        self.saving = False
        end = time.time()
        print(f"MemoryManager: Save operation took {end - start} seconds")

    def push_memory(self, user_id, query, llm_response):
        """Add new memory to the current index for a specific user."""
        start = time.time()
        if user_id not in self.index:
            self.load(user_id)
        doc = Document({"user": query}, {"assistant": llm_response})
        # if not loaded because it didn't exist on disk, then create a new one otherwise just upsert new doc
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

    def pull_memory(self, user_id, query):
        """Fetch memory based on a query for a specific user."""
        if user_id not in self.query_engine:
            print("MemoryManager: Error pull_memory, hash doesn't exists")
            return None
        start = time.time()
        response = self.query_engine[user_id].query(
            query, 
        )
        end = time.time()
        print(f"MemoryManager: pull_memory operation took {end - start} seconds")
        return response

    def delete_memory(self, user_id):
        """Delete all memories for a specific user."""
        userpath = Path(f"{self.dirpath}_{user_id}")
        if userpath.exists() and userpath.is_dir():
            shutil.rmtree(userpath)
        self.index.pop(user_id, None)
        self.query_engine.pop(user_id, None)
