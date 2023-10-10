import time
import datetime
import os
import random
import logging
import traceback
import cachetools.func

from dotenv import load_dotenv
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from typing import List
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field
from langchain.vectorstores import Qdrant
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from cohere_rerank import CohereRerank
from langchain.schema import Document
from datetime import datetime, timedelta
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType


class HTMLItem(BaseModel):
    source_url: str
    html_doc: str

class CacheHTML(BaseModel):
    hash: str

class HTMLInput(BaseModel):
    api_key: str
    action_items: List[HTMLItem] = Field(..., example=[
                                         {"source_url": "http://example.com", "html_doc": "text1"}])
    hash: str
    query: str
    def __str__(self):
        return self.hash + self.query

    def __eq__(self,other):
        return self.hash == other.hash and self.query == other.query

    def __hash__(self):
        return hash(str(self))


class WebManager:

    def __init__(self, rate_limiter):
        load_dotenv()  # Load environment variables
        os.getenv("COHERE_API_KEY")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.collection_name = "web"
        self.client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        self.rate_limiter = rate_limiter

    def create_new_web_retriever(self, api_key: str):
        """Create a new vector store retriever unique to the agent."""
        # create collection if it doesn't exist (if it exists it will fall into finally)
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size = 1536,
                    distance = rest.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(self.collection_name, "metadata.hash_key", field_schema=PayloadSchemaType.KEYWORD)
        except:
            logging.info("WebManager: loaded from cloud...")
        finally:
            logging.info(f"WebManager: Creating memory store with collection {self.collection_name}")
            vectorstore = Qdrant(self.client, self.collection_name, OpenAIEmbeddings(openai_api_key=api_key))
            compressor = CohereRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                    rate_limiter=self.rate_limiter, collection_name=self.collection_name, client=self.client, vectorstore=vectorstore,
                )
            )
            return compression_retriever

    def extract_text_and_source_url(self, retrieved_nodes):
        result = []
        seen = set()
        for document in retrieved_nodes:
            text = document.page_content
            source_url = document.metadata.get('source_url')
            # Create a tuple of text and source_url to check for duplicates
            key = (text, source_url)
            if key not in seen:
                result.append({'text': text, 'source_url': source_url})
                seen.add(key)
        return result

    async def get_retrieved_nodes(self, memory: ContextualCompressionRetriever, function_input: HTMLInput):
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.hash_key", 
                    match=rest.MatchValue(value=function_input.hash), 
                )
            ]
        )
        result = await memory.aget_relevant_documents(function_input.query, filter=filter)
        return result

    @cachetools.func.ttl_cache(maxsize=16384, ttl=36000)
    def load(self, api_key: str):
        """Load existing index data from the filesystem."""
        start = time.time()
        memory = self.create_new_web_retriever(api_key)
        end = time.time()
        logging.info(f"WebManager: Load operation took {end - start} seconds")
        return memory

    async def search_html(self, function_input: HTMLInput):
        """Fetch HTML data based on a query for a specific hash."""
        start = time.time()
        response = []
        nowStamp = datetime.now().timestamp()
        try:
            memory = self.load(function_input.api_key)
            documents = []
            if len(function_input.action_items) > 0:
                hashExist, _ = self.does_hash_exist(function_input.hash)
                if hashExist:
                    function_input.action_items = []
            for item in function_input.action_items:
                text_splitter = SentenceSplitter()
                chunks = text_splitter.split_text(text=item.html_doc)
                documents.extend([Document(page_content=chunk, metadata={"id": random.randint(0, 2**32 - 1), "hash_key": function_input.hash, "last_accessed_at": nowStamp, 'source_url': item.source_url}) for chunk in chunks])
            if len(documents) > 0:
                ids = [doc.metadata["id"] for doc in documents]
                async with self.rate_limiter:
                    await memory.base_retriever.vectorstore.aadd_documents(documents, ids=ids)
                end = time.time()
                logging.info(f"WebManager: Loaded from documents operation took {end - start} seconds")
            nodes = await self.get_retrieved_nodes(memory, function_input)
            response = self.extract_text_and_source_url(nodes)
            # update last_accessed_at
            if len(function_input.action_items) == 0 and len(nodes) > 0:
                ids = [doc.metadata["id"] for doc in nodes]
                for doc in nodes:
                    doc.metadata.pop('relevance_score', None)
                async with self.rate_limiter:
                    await memory.base_retriever.vectorstore.aadd_documents(nodes, ids=ids)
                    self.prune_web()
        except Exception as e:
            logging.warn(f"WebManager: search_html exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(
                f"WebManager: search_html operation took {end - start} seconds")
            return response, end - start

    def prune_web(self):
        """Prune points that are older than 4 hours."""
        current_time = datetime.now()
        one_hour_ago = current_time - timedelta(hours=4)
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.last_accessed_at", 
                    range=rest.Range(lte=one_hour_ago.timestamp()), 
                )
            ]
        )
        self.client.delete(collection_name=self.collection_name, points_selector=filter)

    def does_hash_exist(self, hash: str):
        start = time.time()
        try:
            filter = rest.Filter(
                must=[
                        rest.FieldCondition(
                            key="metadata.hash_key", 
                            match=rest.MatchValue(value=hash), 
                        )
                    ]
            )
            result, _ = self.client.scroll(collection_name=self.collection_name, scroll_filter=filter, limit = 1)
        except Exception as e:
            logging.warn(f"WebManager: does_hash_exist exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
        logging.info(
            f"WebManager: does_hash_exist operation took {end - start} seconds")
        return result is not None and len(result) > 0, end - start
