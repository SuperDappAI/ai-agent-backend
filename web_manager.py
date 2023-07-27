import time
from dotenv import load_dotenv
# from llama_index.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from reader_writer_lock import ReaderWriterLock
import os
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import FAISS
from langchain.retrievers.web_research import WebResearchRetriever
    
class WebManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable
        # Search 
        os.getenv("GOOGLE_CSE_ID")
        os.getenv("GOOGLE_API_KEY")
        self.search = GoogleSearchAPIWrapper()
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
        self.locks = {}  # Dictionary to store locks for each hash

    def get_hash_lock(self, hash_key):
        return self.locks.setdefault(hash_key, ReaderWriterLock())

    def relevance_score_fn(score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # This will differ depending on a few things:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
        # This function converts the euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - score / math.sqrt(2)

    def create_new_web_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        # Define your embedding model
    
        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            self.embeddings.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
        )
        
        return WebResearchRetriever.from_llm(
            vectorstore=vectorstore,
            llm=self.llm, 
            search=self.search, 
        )
        
    def load(self, hash_key):
        """Load existing index data from the filesystem for a specific hash."""
        start = time.time()
        lock = self.get_hash_lock(hash_key)
        lock.writer_acquire()
        try:
            print(f"WebManager: Creating web retriever for {hash_key}")
            self.web_research_retriever[hash_key] = self.create_new_web_retriever()
        finally:
            lock.writer_release()
            end = time.time()
            print(f"WebManager: Operation took {end - start} seconds")

    def search_web(self, hash_key, query):
        """Fetch HTML data based on a query for a specific hash."""
        start = time.time()
        self.load(hash_key)
        lock = self.get_hash_lock(hash_key)
        lock.reader_acquire()
        response = None
        try:
            if hash_key in self.web_research_retriever:
                response = self.web_research_retriever[hash_key].get_relevant_documents(query)
                self.web_research_retriever.pop(hash_key, None)
        finally:
            lock.reader_release()
            end = time.time()
            print(f"WebManager: search_web operation took {end - start} seconds")
            return response, {end - start}