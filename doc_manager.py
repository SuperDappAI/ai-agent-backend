import time
import datetime
import schedule
import os
import asyncio
import random
import requests
import logging
import traceback

from dotenv import load_dotenv
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from qdrant_client import QdrantClient
from pydantic import BaseModel
from langchain.vectorstores import Qdrant
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document
from datetime import datetime
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType

class CacheDoc(BaseModel):
    source_url: str

class DocAddInput(BaseModel):
    source_url: str
    html_doc: str
    category: str

class DocSearchInput(BaseModel):
    query: str
    category: str
    
class DocManager:
    scheduler = schedule.Scheduler()

    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")
        os.getenv("COHERE_API_KEY")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.embeddings = OpenAIEmbeddings()
        self.retriever = None
        self.load()

    def create_new_web_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        # create collection if it doesn't exist (if it exists it will fall into finally)
        collection_name = "doc"
        try:
            client.create_collection(
                on_disk_payload=True,
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size = 1536,
                    distance = rest.Distance.COSINE,
                ),
            )
            client.create_payload_index(collection_name, "metadata.extra_index", field_schema=PayloadSchemaType.KEYWORD)
        except:
            logging.info("DocManager: loaded from disk...")
        finally:
            logging.info(f"DocManager: Creating memory store with collection {collection_name}")
            vectorstore = Qdrant(client, collection_name, self.embeddings)
            compressor = CohereRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                    collection_name=collection_name, client=client, vectorstore=vectorstore,
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

    def get_retrieved_nodes(self, function_input: DocSearchInput):
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.extra_index", 
                    match=rest.MatchValue(value=function_input.category), 
                )
            ]
        )
        result = self.retriever.get_relevant_documents(function_input.query, filter=filter)
        return result

    def load(self):
        """Load existing index data from the filesystem ."""
        start = time.time()
        self.retriever = self.create_new_web_retriever()
        end = time.time()
        logging.info(f"DocManager: Load operation took {end - start} seconds")

    async def add_doc(self, function_input: DocAddInput):
        start = time.time()
        if len(function_input.source_url) <= 0 or len(function_input.html_doc) <= 0:
            logging.warn("DocManager: Cannot add information because data missing")
            end = time.time()
            return "fail", end - start
        srcExist, _ = self.does_source_exist(CacheDoc(source_url=function_input.source_url))
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
            asyncio.create_task(self.retriever.base_retriever.vectorstore.aadd_documents(documents, ids=ids, wait = False))
            end = time.time()
            logging.info(f"DocManager: Loaded from documents operation took {end - start} seconds")
        return "success", end - start

    async def search_doc(self, function_input: DocSearchInput):
        """Fetch Doc data based on a query."""
        start = time.time()
        response = None
        try:
            nodes = self.get_retrieved_nodes(function_input)
            response = self.extract_text_and_source_url(nodes)
            if len(nodes) > 0:
                ids = [doc.metadata["id"] for doc in nodes]
                for doc in nodes:
                    doc.metadata.pop('relevance_score', None)
                asyncio.create_task(self.retriever.base_retriever.vectorstore.aadd_documents(nodes, ids=ids, wait = False))
        except Exception as e:
            logging.warn(f"DocManager: search_html exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(
                f"DocManager: search_html operation took {end - start} seconds")
            return response, end - start

    # async def fetch_web_content(self, url_to_fetch: str, category: str, source_url: str, input_format: str):
    async def fetch_web_content(self, url_to_fetch: str, category: str, source_url: str):
        # fetch md or html from source_url from the web using requests library
        start = time.time()
        response = None
        try:
            response = requests.get(url_to_fetch)
            if response.status_code == 200:
                response = response.content.decode('utf-8')
                doc_add_input = DocAddInput(source_url=source_url, html_doc=response, category=category)
            else:
                logging.warn(f"DocManager: fetch_web_content status code {response.status_code}")
        except:
            logging.warn(f"DocManager: fetch_web_content exception")

        return await self.add_doc(doc_add_input), response, time.time() - start

    def does_source_exist(self, cache_html: CacheDoc):
        start = time.time()
        try:
            result = self.retriever.base_retriever.does_key_exist("metadata.source_url", cache_html.source_url)
        except Exception as e:
            logging.warn(f"DocManager: does_hash_exist exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
        logging.info(
            f"DocManager: does_hash_exist operation took {end - start} seconds")
        return result, end - start
