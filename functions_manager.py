import time
from dotenv import load_dotenv
from llama_index import ServiceContext, VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from llama_index.indices.postprocessor import LLMRerank
from reader_writer_lock import ReaderWriterLock
import sys
import os
import tiktoken
import schedule
import threading
from typing import List
from pathlib import Path
import json
from pydantic import BaseModel, Field
from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.query.schema import QueryBundle


class ActionItem(BaseModel):
    action: str
    intent: str
    category: str


class FunctionInput(BaseModel):
    action_items: List[ActionItem] = Field(..., example=[
                                           {"action": "action_example", "intent": "intent_example", "category": "category_example"}])
    num_results: int = Field(..., example=5)
    similarity_threshold: float = Field(..., example=0.8)


class FunctionsManager1:
    scheduler = schedule.Scheduler()

    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = Path("./storage_functions")
        self.index = None
        self.max_length_allowed = 512
        self.retriever = None
        self.lock = ReaderWriterLock()
        self.reranker = LLMRerank(choice_batch_size=10, top_n=3,
                                  service_context=ServiceContext.from_defaults(
                                      llm=OpenAI(temperature=0,
                                                 model="gpt-3.5-turbo-0613"),
                                  ))
        self.load()

    def stop(self):
        """Stops the scheduler thread."""
        self.save()
        self.stop_event.set()
        self.scheduler_thread.join()

    def transform(self, data, category):
        """Transforms function data for a specific category."""
        result = []
        for item in data:
            page_content = {'name': item['name'], 'category': category, 'description': str(
                item['description'])}
            lenData = len(str(page_content))
            if lenData > self.max_length_allowed:
                print(
                    f"FunctionsManager: transform tried to create a function that surpasses the maximum length allowed max_length_allowed: {self.max_length_allowed} vs length of data: {lenData}")
                continue
            result.append(page_content)
        return result

    def save(self):
        """Persist current index data to the filesystem."""
        start = time.time()
        self.index.storage_context.persist(persist_dir=self.dirpath)
        end = time.time()
        print(f"FunctionsManager: Save operation took {end - start} seconds")

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

    def extract_name_and_category(self, nodes_with_scores):
        result = []
        for node_with_score in nodes_with_scores:
            # Parse the string into a Python dict
            text = json.loads(node_with_score.node.text)
            name = text.get('name')
            category = text.get('category')
            result.append({'name': name, 'category': category})
        return result

    def pull_functions(self, function_input: FunctionInput):
        """Fetch functions based on a query."""
        start = time.time()
        self.lock.reader_acquire()
        response = []
        try:
            for action_item in function_input.action_items:
                query = f"action: {action_item.action} intent: {action_item.intent} category: {action_item.category}"
                parsed_response = self.extract_name_and_category(
                    self.get_retrieved_nodes(query))
                response.append(parsed_response)
        except Exception as e:
            print("Error Exception: " + str(e))
        finally:
            self.lock.reader_release()
            end = time.time()
            return response, end-start

    def get_retrieved_nodes(self, query_str: str):
        query_bundle = QueryBundle(query_str)
        retrieved_nodes = self.retriever.retrieve(query_bundle)
        retrieved_nodes[:] = [
            node for node in retrieved_nodes if node.score >= 0.6]
        # rerank if we need to up to the top_n
        if len(retrieved_nodes) > self.reranker._top_n:
            retrieved_nodes[:] = self.reranker.postprocess_nodes(
                retrieved_nodes, query_bundle)
        return retrieved_nodes

    def load(self):
        """Load existing index data from the filesystem."""
        start = time.time()
        result = False
        try:
            if self.dirpath.exists() and self.dirpath.is_dir():
                print("FunctionsManager: Loading from disk")
                # rebuild storage context
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.dirpath)
                # load index
                self.index = load_index_from_storage(storage_context)
                self.retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=10
                )
                result = True
            else:
                # Try to load index data from the filesystem only if not running tests
                if 'unittest' not in sys.modules.keys():
                    # If loading was unsuccessful (e.g., no data on the filesystem), load functions from JSON file
                    with open('./utils/functions.json', 'r') as f:
                        print("FunctionsManager: Loading from functions.json")
                        functions_json = json.load(f)
                        self.push_functions(functions_json)
                        result = True
        except Exception as e:
            print("Error Exception: " + str(e))
        finally:
            end = time.time()
            print(f"FunctionsManager: Load took {end - start} seconds")
            return result

    def push_functions(self, functions):
        """Update the current index with new functions."""
        start = time.time()
        self.lock.writer_acquire()
        tokens = None
        try:
            print("FunctionsManager: adding functions to index...")

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

            documents = [Document(text=json.dumps(t)) for t in all_docs]
            self.index = VectorStoreIndex.from_documents(documents)
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=10
            )
            tokens = self.count_tokens(functions)
            self.save()
        except Exception as e:
            print("Error Exception: " + str(e))
        finally:
            self.lock.writer_release()
            end = time.time()
            print(
                f"FunctionsManager: push_functions took {end - start} seconds")
            return tokens, end-start
