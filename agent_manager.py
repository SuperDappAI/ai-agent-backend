import time
from dotenv import load_dotenv
from pathlib import Path
import os
from datetime import datetime
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from time_weighted_retriever import TimeWeightedVectorStoreRetriever
from generative_memory import GenerativeAgentMemory
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType

class AgentManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = "./storage_memory"
        self.embeddings = OpenAIEmbeddings()
        self.memory = {}
        self.payload_conversation_index_key = "metadata.conversation"
        self.LLM = OpenAI()

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
                client=client, vectorstore=vectorstore, decay_rate=0.001, search_kwargs={"score_threshold":0.72}, other_score_keys=["importance"], k=15
            )

    def create_memory(self, user_id, path):
        return GenerativeAgentMemory(
            llm=self.LLM,
            memory_retriever=self.create_new_memory_retriever(user_id, path),
            verbose=False
        )

    def load(self, user_id):
        """Load existing index data from the filesystem for a specific user."""
        start = time.time()
        userpath = Path(f"{self.dirpath}/{user_id}")
        self.memory[user_id] = self.create_memory(user_id, userpath)
        end = time.time()
        print(f"AgentManager: Load operation took {end - start} seconds")

    def push_memory(self, user_id, conversation_id, query, llm_response):
        """Add new memory to the current index for a specific user."""
        start = time.time()
        if user_id not in self.memory:
            self.load(user_id)
        try:
            self.memory[user_id].save_context(
                {
                    self.memory[user_id].add_memory_user_key: query,
                    self.memory[user_id].add_memory_aida_key: llm_response,
                    self.memory[user_id].now_key: datetime.now(),
                    self.memory[user_id].payload_conversation_key: conversation_id,
                },
            )
        except Exception as e:
            print(f"AgentManager: push_memory exception {e}") 
        finally:
            end = time.time()
            print(f"AgentManager: push_memory operation took {end - start} seconds")
            return end - start

    def pull_memory(self, user_id, convo_id, query):
        """Fetch memory based on a query for a specific user."""
        start = time.time()
        if user_id not in self.memory:
            self.load(user_id)
        response = None
        try:
            if user_id in self.memory:
                response = self.memory[user_id].load_memory_variables(
                {
                    self.memory[user_id].queries_key: [query],
                    self.memory[user_id].payload_conversation_key: convo_id,
                }
            )
        except Exception as e:
            print(f"AgentManager: pull_memory exception {e}")
        finally:
            end = time.time()
            print(f"AgentManager: pull_memory operation took {end - start} seconds")
            return response, end - start

    def clear_conversation(self, user_id, conversation_id):
        """Delete all memories for a specific conversation with a user."""
        start = time.time()
        try:
            filter = {self.payload_conversation_index_key: conversation_id}
            qdrant_filter = self.memory[user_id].vectorstore._qdrant_filter_from_dict(filter)
            self.memory[user_id].vectorstore._build_condition(self.payload_conversation_index_key, conversation_id)
            self.memory[user_id].client.clear_payload(collection_name=user_id, points=qdrant_filter, wait = False)
            self.memory[user_id].client.delete_vectors(collection_name=user_id, points=qdrant_filter, wait = False)
        finally:
            end = time.time()
            print(f"AgentManager: delete_memory operation took {end - start} seconds")
            return "success", end - start