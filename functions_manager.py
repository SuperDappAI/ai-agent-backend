import time
from dotenv import load_dotenv
import tiktoken
import schedule
from qdrant_client import QdrantClient
from typing import List
import json
import asyncio
import threading
import os
import uuid
import logging
from datetime import datetime
from pydantic import BaseModel, Field
from qdrant_client.http import models as rest
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document
from datetime import datetime, timedelta


class ActionItem(BaseModel):
    action: str
    intent: str
    category: str


class FunctionInput(BaseModel):
    action_items: List[ActionItem] = Field(..., example=[
                                           {"action": "action_example", "intent": "intent_example", "category": "category_example"}])
    num_semantic_results: int = Field(..., example=10)
    similarity_threshold: float = Field(..., example=0.72)


class FunctionsManager1:
    scheduler = schedule.Scheduler()

    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        os.getenv("COHERE_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.embeddings = OpenAIEmbeddings()
        self.index = None
        self.max_length_allowed = 512
        self.retriever = None
        self.scheduler.every(2).weeks.do(self.prune_functions)

        # Create new thread for schedule
        self.stop_event = threading.Event()
        self.scheduler_thread = threading.Thread(target=self.run_continuously)
        self.scheduler_thread.start()

    def run_continuously(self):
        """Keep checking and running pending tasks every second."""
        while not self.stop_event.is_set():
            self.scheduler.run_pending()
            time.sleep(1)

    def stop(self):
        """Stops the scheduler thread."""
        self.stop_event.set()
        self.scheduler_thread.join()

    def create_new_functions_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        collection_name = "functions"
        client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        was_created = False
        # create collection if it doesn't exist (if it exists it will fall into finally)
        try:
            client.create_collection(
                on_disk_payload=True,
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=1536,
                    distance=rest.Distance.COSINE,
                ),
            )
            was_created = True
        except Exception as e:
            logging.warn(
                f"FunctionsManager: create_new_functions_retriever exception {e}")
        finally:
            logging.info(
                f"FunctionsManager: Creating memory store with collection {collection_name}")
            vectorstore = Qdrant(client, collection_name, self.embeddings)
            compressor = CohereRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                    collection_name=collection_name, client=client, vectorstore=vectorstore,
                )
            )
            return was_created, compression_retriever

    def transform(self, data, category):
        """Transforms function data for a specific category."""
        now = datetime.now().timestamp()
        result = []
        for item in data:
            page_content = {'name': item['name'], 'category': category, 'description': str(
                item['description'])}
            lenData = len(str(page_content))
            if lenData > self.max_length_allowed:
                logging.info(
                    f"FunctionsManager: transform tried to create a function that surpasses the maximum length allowed max_length_allowed: {self.max_length_allowed} vs length of data: {lenData}")
                continue
            metadata = {
                "id":  uuid.uuid4().hex,
                "extra_index": category,
                "last_accessed_at": now,
            }
            doc = Document(
                page_content=json.dumps(page_content),
                metadata=metadata
            )
            result.append(doc)
        return result

    def count_tokens(self, functions):
        """Count the tokens for all the functions."""
        function_types = ['information_retrieval',
                          'communication',
                          'data_processing',
                          'sensory_perception']

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        tokens = []
        for func_type in function_types:
            if func_type in functions:
                for func in functions[func_type]:
                    function_string = json.dumps(func)
                    tokens.append(
                        {func['name']: len(encoding.encode(function_string))})
        return tokens

    def extract_name_and_category(self, documents):
        result = []
        seen = set()  # Track seen combinations of name and category
        for doc in documents:
            # Parse the page_content string into a Python dict
            text = json.loads(doc.page_content)
            name = text.get('name')
            category = text.get('category')

            # Check if this combination has been seen before
            if (name, category) not in seen:
                result.append({'name': name, 'category': category})
                seen.add((name, category))  # Mark this combination as seen

        return result

    async def pull_functions(self, function_input: FunctionInput):
        """Fetch functions based on a query."""
        start = time.time()
        response = []
        try:
            for action_item in function_input.action_items:
                query = f"action: {action_item.action} intent: {action_item.intent} category: {action_item.category}"
                documents = self.get_retrieved_nodes(
                    query, action_item.category, function_input.similarity_threshold, function_input.num_semantic_results)
                if len(documents) > 0:
                    parsed_response = self.extract_name_and_category(documents)
                    response.append(parsed_response)
                    ids = [doc.metadata["id"] for doc in documents]
                    for doc in documents:
                        doc.metadata.pop('relevance_score', None)
                    asyncio.create_task(self.retriever.base_retriever.vectorstore.aadd_documents(documents, ids=ids, wait = False))
        except Exception as e:
            logging.warn(f"FunctionsManager: pull_functions exception {e}")
        finally:
            end = time.time()
            logging.info(
                f"FunctionsManager: pull_functions operation took {end - start} seconds")
            return response, end-start

    def get_retrieved_nodes(self, query_str: str, category: str, score: float, num_semantic_results: int):
        kwargs = {"extra_index": category,
                "score_threshold": score, "k": num_semantic_results}
        return self.retriever.get_relevant_documents(query_str, **kwargs)

    async def load(self):
        """Load existing index data from the filesystem for a specific user."""
        start = time.time()
        was_created, self.retriever = self.create_new_functions_retriever()
        logging.info(
            f"FunctionsManager: load create_new_functions_retriever was_created? {was_created}")
        if was_created:
            logging.info(
                f"FunctionsManager: unittest not in sys.modules.keys()")
            # If loading was unsuccessful (e.g., no data on the filesystem), load functions from JSON file
            with open('./utils/functions.json', 'r') as f:
                print("FunctionsManager: Loading from functions.json")
                functions_json = json.load(f)
                await self.push_functions(functions_json)
        end = time.time()
        logging.info(
            f"FunctionsManager: Load operation took {end - start} seconds")

    async def push_functions(self, functions):
        """Update the current index with new functions."""
        start = time.time()
        tokens = None
        try:
            logging.info("FunctionsManager: adding functions to index...")

            function_types = ['information_retrieval',
                              'communication',
                              'data_processing',
                              'sensory_perception']

            all_docs = []

            # Transform and concatenate function types
            for func_type in function_types:
                if func_type in functions:
                    transformed_functions = self.transform(
                        functions[func_type], func_type.replace('_', ' ').title())
                    all_docs.extend(transformed_functions)
            ids = [doc.metadata["id"] for doc in all_docs]
            await self.retriever.base_retriever.vectorstore.aadd_documents(all_docs, ids=ids)
            tokens = self.count_tokens(functions)
        except Exception as e:
            logging.warn(f"FunctionsManager: push_functions exception {e}")
        finally:
            end = time.time()
            logging.info(
                f"FunctionsManager: push_functions took {end - start} seconds")
            return tokens, end-start

    def prune_functions(self):
        """Prune functions that haven't been used for atleast six weeks."""
        current_time = datetime.now()
        one_hour_ago = current_time - timedelta(weeks=6)
        self.retriever.base_retriever.prune_from(one_hour_ago.timestamp())
