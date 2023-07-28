import time
import shutil
from dotenv import load_dotenv
from reader_writer_lock import ReaderWriterLock
from pathlib import Path
import schedule
import threading
import os
import faiss
import json
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain_experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
class AgentManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./storage_memory"
        self.embeddings = OpenAIEmbeddings()
        self.agent = {}
        self.locks = {} 
        self.LLM = ChatOpenAI()

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

    def create_new_memory_retriever(self, vectorstore):
        """Create a new vector store retriever unique to the agent."""
        # Define your embedding model
        if vectorstore is None:
            # Initialize the vectorstore as empty
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)
            vectorstore = FAISS(
                self.embeddings.embed_query,
                index,
                InMemoryDocstore({}),
                {}
            )
        
        return TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, other_score_keys=["importance"], k=15
        )
    def create_agent(self, faiss_vectorstore, user_id):
        memory = GenerativeAgentMemory(
            llm=self.LLM,
            memory_retriever=self.create_new_memory_retriever(faiss_vectorstore),
            verbose=False,
        )
        return GenerativeAgent(
            name="AiDA",
            age=25,
            traits="helpful, courteous, curious",
            status=f"Personal assistant to {user_id}",
            llm=self.LLM,
            memory=memory,
        )

    def load(self, user_id):
        """Load existing index data from the filesystem for a specific user."""
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            start = time.time()
            userpath = Path(f"{self.dirpath}/{user_id}.faiss")
            if userpath.exists():
                print("AgentManager: Loading from disk")
                self.agent[user_id] = self.create_agent(FAISS.load_local(self.dirpath,self.embeddings,user_id), user_id)
            else:
                print("AgentManager: Creating from scratch")
                self.agent[user_id] = self.create_agent(None, user_id)
                lock.dirty = True
            end = time.time()
            print(f"AgentManager: Load operation took {end - start} seconds")
        finally:
            lock.writer_release()

    def save(self):
        """Persist current index data to the filesystem."""
        for user_id, idx in self.agent.items():
            lock = self.get_user_lock(user_id)
            lock.writer_acquire()
            try:
                start = time.time()
                if lock.dirty:
                    #idx.memory.memory_retriever.vectorstore.save_local(self.dirpath, user_id)
                    lock.dirty = False
                end = time.time()
                print(f"AgentManager: Save operation for user {user_id} took {end - start} seconds")
            finally:
                lock.writer_release()

    def push_memory(self, user_id, query, llm_response):
        """Add new memory to the current index for a specific user."""
        start = time.time()
        if user_id not in self.agent:
            self.load(user_id)
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            obj = {"user": query, "AiDA": llm_response}
            self.agent[user_id].memory.add_memory(json.dumps(obj))
            lock.dirty = True
        finally:
            lock.writer_release()
            end = time.time()
            print(f"AgentManager: push_memory operation took {end - start} seconds")
            return end - start

    def pull_memory(self, user_id, query):
        """Fetch memory based on a query for a specific user."""
        start = time.time()
        if user_id not in self.agent:
            self.load(user_id)
        lock = self.get_user_lock(user_id)
        lock.reader_acquire()
        response = None
        try:
            if user_id in self.agent:
                response = self.agent[user_id].memory.fetch_memories(query, now=datetime.now())
        except Exception as e:
            print(f"AgentManager: pull_memory exception {e}")
        finally:
            lock.reader_release()
            end = time.time()
            print(f"AgentManager: pull_memory operation took {end - start} seconds")
            return response, {end - start}


    def delete_memory(self, user_id):
        """Delete all memories for a specific user."""
        start = time.time()
        lock = self.get_user_lock(user_id)
        lock.writer_acquire()
        try:
            userpath = Path(f"{self.dirpath}/{user_id}")
            if userpath.exists() and userpath.is_dir():
                shutil.rmtree(userpath)
            self.agent.pop(user_id, None)
        finally:
            lock.writer_release()
            end = time.time()
            return {end - start}