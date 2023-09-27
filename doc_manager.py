import time
import datetime
import schedule
import os
import asyncio
import random
import logging
import traceback
import cachetools.func

from dotenv import load_dotenv
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from qdrant_client import QdrantClient
from pydantic import BaseModel
from langchain.vectorstores import Qdrant
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from cohere_rerank import CohereRerank
from langchain.schema import Document
from datetime import datetime
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType

class CacheDoc(BaseModel):
    source_url: str

class DocAddInput(BaseModel):
    api_key: str
    source_url: str
    html_doc: str
    category: str

class DocSearchInput(BaseModel):
    api_key: str
    query: str
    category: str
    def __str__(self):
        return self.query + self.category

    def __eq__(self,other):
        return self.query == other.query and self.category == other.category

    def __hash__(self):
        return hash(str(self))    
    
class DocManager:
    scheduler = schedule.Scheduler()

    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("COHERE_API_KEY")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        self.collection_name = "doc"

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
            self.client.create_payload_index(self.collection_name, "metadata.extra_index", field_schema=PayloadSchemaType.KEYWORD)
        except:
            logging.info("DocManager: loaded from cloud...")
        finally:
            logging.info(f"DocManager: Creating memory store with collection {self.collection_name}")
            vectorstore = Qdrant(self.client, self.collection_name, OpenAIEmbeddings(openai_api_key=api_key))
            compressor = CohereRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                    collection_name=self.collection_name, client=self.client, vectorstore=vectorstore,
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

    async def get_retrieved_nodes(self, memory: ContextualCompressionRetriever, function_input: DocSearchInput):
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.extra_index", 
                    match=rest.MatchValue(value=function_input.category), 
                )
            ]
        )
        result = await memory.aget_relevant_documents(function_input.query, filter=filter)
        return result

    @cachetools.func.lru_cache(maxsize=16384)
    def load(self, api_key: str):
        """Load existing index data from the filesystem ."""
        start = time.time()
        memory = self.create_new_web_retriever(api_key)
        end = time.time()
        logging.info(f"DocManager: Load operation took {end - start} seconds")
        return memory

    async def add_doc(self, function_input: DocAddInput):
        start = time.time()
        if len(function_input.source_url) <= 0 or len(function_input.html_doc) <= 0:
            logging.warn("DocManager: Cannot add information because data missing")
            end = time.time()
            return "fail", end - start
        memory = self.load(function_input.api_key)
        srcExist, _ = self.does_source_exist(function_input.source_url)
        if srcExist:
            logging.warn("DocManager: source_url already exists")
            end = time.time()
            return "fail", end - start
        nowStamp = datetime.now().timestamp()
        documents = []
        if len(function_input.html_doc) > 0:
            text_splitter = SentenceSplitter()
            chunks = text_splitter.split_text(text=function_input.html_doc)
            documents.extend([Document(page_content=chunk, metadata={"id": random.randint(0, 2**32 - 1), "extra_index": function_input.category, "last_accessed_at": nowStamp, 'source_url': function_input.source_url}) for chunk in chunks])
        if len(documents) > 0:
            ids = [doc.metadata["id"] for doc in documents]
            asyncio.create_task(memory.base_retriever.vectorstore.aadd_documents(documents, ids=ids, wait = False))
            end = time.time()
            logging.info(f"DocManager: Loaded from documents operation took {end - start} seconds")
        return "success", end - start

    async def search_doc(self, function_input: DocSearchInput):
        """Fetch Doc data based on a query."""
        start = time.time()
        response = None
        try:
            memory = self.load(function_input.api_key)
            nodes = await self.get_retrieved_nodes(memory, function_input)
            response = self.extract_text_and_source_url(nodes)
            # update last_accessed_at
            if len(nodes) > 0:
                ids = [doc.metadata["id"] for doc in nodes]
                for doc in nodes:
                    doc.metadata.pop('relevance_score', None)
                asyncio.create_task(memory.base_retriever.vectorstore.aadd_documents(nodes, ids=ids, wait = False))
        except Exception as e:
            logging.warn(f"DocManager: search_html exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(
                f"DocManager: search_html operation took {end - start} seconds")
            return response, end - start

    def does_source_exist(self, source_url: str):
        start = time.time()
        try:
            filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.source_url", 
                        match=rest.MatchValue(value=source_url), 
                    )
                ]
            )
            result, _ = self.client.scroll(collection_name=self.collection_name, scroll_filter=filter, limit = 1)
        except Exception as e:
            logging.warn(f"DocManager: does_source_exist exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
        logging.info(
            f"DocManager: does_source_exist operation took {end - start} seconds")
        return result is not None and len(result) > 0, end - start
