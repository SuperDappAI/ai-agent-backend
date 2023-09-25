import time
import tiktoken
import json
import asyncio
import os
import random
import logging
import traceback
import cachetools.func

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field
from qdrant_client.http import models as rest
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.retrievers import ContextualCompressionRetriever
from cohere_rerank import CohereRerank
from langchain.schema import Document
from datetime import datetime, timedelta
from qdrant_client.http.models import PayloadSchemaType

class ActionItem(BaseModel):
    action: str
    intent: str
    category: str
    def __str__(self):
        return self.action + self.intent + self.category

    def __eq__(self,other):
        return self.action == other.action and self.intent == other.intent and self.category == other.category

    def __hash__(self):
        return hash(str(self))

class FunctionInput(BaseModel):
    api_key: str
    user_id: str = None
    action_items: List[ActionItem] = Field(..., example=[
                                           {"action": "action_example", "intent": "intent_example", "category": "category_example"}])
    def __str__(self):
        if self.user_id:
            return self.api_key + str(self.action_items) + self.user_id
        else:
            return self.api_key + str(self.action_items)

    def __eq__(self,other):
        return self.api_key == other.api_key and self.action_items == other.action_items and self.user_id == other.user_id

    def __hash__(self):
        return hash(str(self))

class FunctionItem(BaseModel):
    name: str
    description: str
    category: str

class FunctionOutput(BaseModel):
    api_key: str
    user_id: str = None
    functions: List[FunctionItem] = Field(..., example=[
                                           {"name": "name_example", "description": "description_example", "category": "category_example"}])
    
class FunctionsManager:

    def __init__(self):
        load_dotenv()  # Load environment variables
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        os.getenv("COHERE_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.index = None
        self.max_length_allowed = 512
        self.collection_name = "functions"
        self.client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        self.inited = False
        
    def create_new_functions_retriever(self, api_key: str):
        """Create a new vector store retriever unique to the agent."""
        # create collection if it doesn't exist (if it exists it will fall into finally)
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=1536,
                    distance=rest.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(self.collection_name, "metadata.user_id", field_schema=PayloadSchemaType.KEYWORD)
        except:
            logging.info(f"FunctionsManager: loaded from cloud...")
        finally:
            logging.info(
                f"FunctionsManager: Creating memory store with collection {self.collection_name}")
            vectorstore = Qdrant(self.client, self.collection_name, OpenAIEmbeddings(openai_api_key=api_key))
            compressor = CohereRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                    collection_name=self.collection_name, client=self.client, vectorstore=vectorstore,
                )
            )
            return compression_retriever

    def transform(self, user_id, data, category):
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
                "id":  random.randint(0, 2**32 - 1),
                "user_id": user_id,
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

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
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
        if self.inited is False:
            try:
                self.client.get_collection(self.collection_name)
            except:
                with open('./utils/functions.json', 'r') as f:
                    print("FunctionsManager: Loading from functions.json")
                    functions_json = json.load(f)
                    await self.push_functions(function_input.user_id, function_input.api_key, functions_json)
            self.inited = True
        memory = self.load(function_input.api_key)
        response = []
        #loop = asyncio.get_event_loop()
        try:
            for action_item in function_input.action_items:
                query = f"action: {action_item.action} intent: {action_item.intent} category: {action_item.category}"
                documents = await self.get_retrieved_nodes(memory,
                    query, action_item.category, function_input.user_id)
                if len(documents) > 0:
                    parsed_response = self.extract_name_and_category(documents)
                    response.append(parsed_response)
                    # update last_accessed_at
                    ids = [doc.metadata["id"] for doc in documents]
                    for doc in documents:
                        doc.metadata.pop('relevance_score', None)
                    asyncio.create_task(memory.base_retriever.vectorstore.aadd_documents(documents, ids=ids, wait = False))
                    #loop.run_in_executor(None, self.prune_functions)
        except Exception as e:
            logging.warn(f"FunctionsManager: pull_functions exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(
                f"FunctionsManager: pull_functions operation took {end - start} seconds")
            return response, end-start

    async def get_retrieved_nodes(self, memory: ContextualCompressionRetriever, query_str: str, category: str, user_id: str):
        kwargs = {}
        if len(category) > 0:
            kwargs["extra_index"] = category
        # if user provided then look for null or direct matches, otherwise look for null so it matches public functions
        if user_id:
            filter = rest.Filter(
                should=[
                    rest.FieldCondition(
                        key="metadata.user_id",
                        match=rest.MatchValue(value=user_id),
                    ),
                    rest.IsNullCondition(
                        is_null=rest.PayloadField(key="metadata.user_id")
                    )
                ]
            )
            kwargs["user_filter"] = filter
        else:
            filter = rest.Filter(
                should=[
                    rest.IsNullCondition(
                        is_null=rest.PayloadField(key="metadata.user_id")
                    )
                ]
            )
            kwargs["user_filter"] = filter
        return await memory.aget_relevant_documents(query_str, **kwargs)

    @cachetools.func.ttl_cache(maxsize=16384, ttl=36000)
    def load(self, api_key: str):
        """Load existing index data from the filesystem for a specific user."""
        start = time.time()
        memory = self.create_new_functions_retriever(api_key)
        end = time.time()
        logging.info(
            f"FunctionsManager: Load operation took {end - start} seconds")
        return memory

    async def push_functions(self, user_id: str, api_key: str, functions):
        """Update the current index with new functions."""
        start = time.time()
        tokens = None
        memory = self.load(api_key)
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
                        user_id, functions[func_type], func_type.replace('_', ' ').title())
                    all_docs.extend(transformed_functions)
            ids = [doc.metadata["id"] for doc in all_docs]
            await memory.base_retriever.vectorstore.aadd_documents(all_docs, ids=ids)
            tokens = self.count_tokens(functions)
        except Exception as e:
            logging.warn(f"FunctionsManager: push_functions exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(
                f"FunctionsManager: push_functions took {end - start} seconds")
            return tokens, end-start

    def prune_functions(self):
        """Prune functions that haven't been used for atleast six weeks."""
        def attempt_prune():
            current_time = datetime.now()
            six_weeks_ago = current_time - timedelta(weeks=6)
            filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.last_accessed_at", 
                        range=rest.Range(lte=six_weeks_ago.timestamp()), 
                    )
                ]
            )
            self.client.delete(collection_name=self.collection_name, points_selector=filter, wait = False)
        try:
            attempt_prune()
        except Exception as e:
            logging.warn(f"FunctionsManager: prune_functions exception {e}\n{traceback.format_exc()}")
            # Attempt a second prune after reload
            try:
                attempt_prune()
            except Exception as e:
                # If prune after reload fails, propagate the error upwards
                logging.error(f"FunctionsManager: prune_functions failed after reload, exception {e}\n{traceback.format_exc()}")
                raise
        return True