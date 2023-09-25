import time
import logging
import os
import cachetools.func

from dotenv import load_dotenv
from typing import Any, Dict, List
from document_summarizer import FlexibleDocumentSummarizer
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType
from langchain.retrievers import ContextualCompressionRetriever
from qdrant_retriever import QDrantVectorStoreRetriever
from cohere_rerank import CohereRerank
from langchain.embeddings import OpenAIEmbeddings
from generative_conversation_summarized_memory import GenerativeAgentConversationSummarizedMemory

class MemorySummarizer:
    flexible_document_summarizer: FlexibleDocumentSummarizer
    def __init__(self, flexible_document_summarizer, agent_manager):
        load_dotenv()  # Load environment variables
        os.getenv("COHERE_API_KEY")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.agent_manager = agent_manager
        self.flexible_document_summarizer = flexible_document_summarizer

    def create_new_conversation_summarizer(self, api_key: str, user_id: str):
        """Create a new vector store retriever unique to the agent."""
        collection_name = f"{user_id}_summaries"
        # create collection if it doesn't exist
        try:
            self.agent_manager.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=1536,
                    distance=rest.Distance.COSINE,
                ),
            )
            self.agent_manager.client.create_payload_index(collection_name, "metadata.extra_index", field_schema=PayloadSchemaType.KEYWORD)
        except:
            print("MemorySummarizer: loaded from cloud...")
        finally:
            logging.info(f"MemorySummarizer: Creating memory store with collection {collection_name}")
            vectorstore = Qdrant(self.agent_manager.client, collection_name, OpenAIEmbeddings(openai_api_key=api_key))
            compressor = CohereRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                    collection_name=collection_name, client=self.agent_manager.client, vectorstore=vectorstore,
                )
            )
            return compression_retriever
            
    def create_summarized_memory(self, api_key: str, user_id:str):
        return GenerativeAgentConversationSummarizedMemory(
            llm=ChatOpenAI(openai_api_key=api_key, temperature=0, max_tokens=2048, model="gpt-3.5-turbo"),
            memory_retriever=self.create_new_conversation_summarizer(api_key, user_id),
            verbose=self.agent_manager.verbose
        )

    @cachetools.func.ttl_cache(maxsize=16384, ttl=36000)
    def load(self, api_key: str, user_id:str) -> GenerativeAgentConversationSummarizedMemory:
        """Load existing index data from the cloud."""
        start = time.time()
        retriever = self.create_summarized_memory(api_key, user_id)
        end = time.time()
        logging.info(f"MemorySummarizer: Load operation took {end - start} seconds")
        return retriever

    async def save(self, api_key: str, user_id: str, outputs: Dict[str, Any]) -> List[str]:
        memory = self.load(api_key, user_id)
        await memory.save_context(outputs)