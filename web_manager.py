import time
import datetime
from dotenv import load_dotenv
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from typing import List
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field
import schedule
import threading
import os
import asyncio
from langchain.vectorstores import Qdrant
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document
from datetime import datetime, timedelta
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType
import logging

class HTMLItem(BaseModel):
    source_url: str
    html_doc: str


class HTMLInput(BaseModel):
    action_items: List[HTMLItem] = Field(..., example=[
                                         {"source_url": "http://example.com", "html_doc": "text1"}])
    hash: str
    query: str
    num_semantic_results: int = Field(..., example=10)
    similarity_threshold: float = Field(..., example=0.72)


class WebManager:
    scheduler = schedule.Scheduler()

    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")
        os.getenv("COHERE_API_KEY")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.embeddings = OpenAIEmbeddings()
        
        self.retriever = None
        self.collection_name = "web"
        self.scheduler.every(3600).seconds.do(self.prune_web)

        # Create new thread for schedule
        self.stop_event = threading.Event()
        self.scheduler_thread = threading.Thread(target=self.run_continuously)
        self.scheduler_thread.start()
        self.load()

    def create_new_web_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        # create collection if it doesn't exist (if it exists it will fall into finally)
        try:
            client.create_collection(
                on_disk_payload=True,
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size = 1536,
                    distance = rest.Distance.COSINE,
                ),
            )
            client.create_payload_index(self.collection_name, "metadata.hash_key", field_schema=PayloadSchemaType.KEYWORD)
        except:
            logging.info("FunctionsManager: loaded from disk...")
        finally:
            logging.info(f"FunctionsManager: Creating memory store with collection {self.collection_name}")
            vectorstore = Qdrant(client, self.collection_name, self.embeddings)
            compressor = CohereRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                    collection_name=self.collection_name, client=client, vectorstore=vectorstore,
                )
            )
            return compression_retriever
            

    def run_continuously(self):
        """Keep checking and running pending tasks every second."""
        while not self.stop_event.is_set():
            self.scheduler.run_pending()
            time.sleep(1)

    def stop(self):
        """Stops the scheduler thread."""
        self.stop_event.set()
        self.scheduler_thread.join()

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

    def get_retrieved_nodes(self, function_input: HTMLInput):
        filter_dict = {
            'must': {
                "metadata.hash_key": {
                    'match': {'value': function_input.hash}
                }
            }
        }
        filter = self.retriever.base_retriever._qdrant_filter_from_dict(filter_dict)
        result = self.retriever.base_retriever.get_relevant_documents(function_input.query, filter=filter, score_threshold=function_input.similarity_threshold, k=function_input.num_semantic_results)
        return result

    def load(self):
        """Load existing index data from the filesystem for a specific hash."""
        start = time.time()
        self.retriever = self.create_new_web_retriever()
        end = time.time()
        logging.info(f"WebManager: Load operation took {end - start} seconds")

    async def search_html(self, function_input: HTMLInput):
        """Fetch HTML data based on a query for a specific hash."""
        start = time.time()
        response = None
        nowStamp = datetime.now().timestamp()
        try:
            documents = []
            for item in function_input.action_items:
                text_splitter = SentenceSplitter()
                chunks = text_splitter.split_text(text=item.html_doc)
                documents.extend([Document(page_content=chunk, metadata={"hash_key": function_input.hash, "last_accessed_at": nowStamp, 'source_url': item.source_url}) for chunk in chunks])
            if len(documents) > 0:
                await self.retriever.base_retriever.vectorstore.aadd_documents(documents)
                end = time.time()
                logging.info(f"WebManager: Loaded from documents operation took {end - start} seconds")
            nodes = self.get_retrieved_nodes(function_input)
            response = self.extract_text_and_source_url(nodes)
            if len(documents) > 0:
                for doc, _ in documents:
                    doc.metadata["last_accessed_at"] = nowStamp
                asyncio.create_task(self.retriever.base_retriever.vectorstore.aadd_documents(documents, wait = False))
        except Exception as e:
            logging.warn(f"WebManager: search_html exception {e}")
        finally:
            end = time.time()
            logging.info(
                f"WebManager: search_html operation took {end - start} seconds")
            return response, end - start

    def prune_web(self):
        """Prune cache that are older than an hour."""
        current_time = datetime.now()
        one_hour_ago = current_time - timedelta(hours=1)
        self.retriever.base_retriever.prune_from(one_hour_ago.timestamp())

    def does_hash_exist(self, hash):
        return self.retriever.base_retriever.does_key_exist("metadata.hash_key", hash)
