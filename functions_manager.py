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
from pathlib import Path
import json
from typing import List
from pydantic import BaseModel, Field

class ActionItem(BaseModel):
    action: str
    intent: str
    category: str

class FunctionInput(BaseModel):
    action_items: List[ActionItem] = Field(..., example=[{"action": "action_example", "intent": "intent_example", "category": "category_example"}])
    num_results: int = Field(..., example=5)
    similarity_threshold: float = Field(..., example=0.8)


class FunctionsManager1:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = Path("./storage_functions")
        self.index = None
        self.dirty = False
        self.query_engine = None
        self.lock = ReaderWriterLock()
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, 
            service_context=ServiceContext.from_defaults(
                llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
            ))
        # Try to load index data from the filesystem only if not running tests
        if 'unittest' not in sys.modules.keys() and not self.load():
            # If loading was unsuccessful (e.g., no data on the filesystem), load functions from JSON file
            with open('./utils/functions.json', 'r') as f:
                functions_json = json.load(f)
                self.push_functions(functions_json)
                self.save()

        # Save function scheduled to run every 5 to 10 minutes
        schedule.every(300).to(600).seconds.do(self.save)
        
        # Create new thread for schedule
        self.stop_event = threading.Event()
        self.scheduler_thread = threading.Thread(target=self.run_continuously)
        self.scheduler_thread.start()

    def run_continuously(self):
        """Keep checking and running pending tasks every second."""
        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

    def stop(self):
        """Stops the scheduler thread."""
        self.save()
        self.stop_event.set()
        self.scheduler_thread.join()

    def transform(self, data, category):
        """Transforms function data for a specific category."""
        result = []
        for item in data:
            page_content = {'name': item['name'], 'category': category, 'description': str(item['description'])}
            result.append(page_content)
        return result

    def save(self):
        """Persist current index data to the filesystem."""
        start = time.time()
        self.lock.writer_acquire()
        try:
            if self.dirty is True:
                self.index.storage_context.persist(persist_dir=self.dirpath)
                self.dirty = False
        finally:
            self.lock.writer_release()
            end = time.time()
            print(f"FunctionsManager: Save operation took {end - start} seconds")

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
                    tokens.append({func['name']: len(encoding.encode(function_string))})
        return tokens

    def pull_functions(self, function_input: FunctionInput):
        """Fetch functions based on a query."""
        start = time.time()
        self.lock.reader_acquire()
        response = []
        try:
            if self.query_engine is not None:
                for action_item in function_input.action_items:
                    query = f"action: {action_item.action} intent: {action_item.intent} category: {action_item.category}"
                    print(query)
                    response.append(self.query_engine.query(query))
        finally:
            self.lock.reader_release()
            end = time.time()
            return response, {end-start}


    def load(self):
        """Load existing index data from the filesystem."""
        start = time.time()
        self.lock.writer_acquire()
        result = False
        try:
            print("FunctionsManager: Loading from disk")
            if self.dirpath.exists() and self.dirpath.is_dir():
                # rebuild storage context
                storage_context = StorageContext.from_defaults(persist_dir=self.dirpath)

                # load index
                self.index = load_index_from_storage(storage_context)
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=10,
                    node_postprocessors=[self.reranker]
                )
                result = True
        finally:
            self.lock.writer_release()
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
                    transformed_functions = self.transform(functions[func_type], func_type.replace('_', ' ').title())
                    all_docs.extend(transformed_functions)

            
            documents = [Document(text=json.dumps(t)) for t in all_docs]
            self.index = VectorStoreIndex.from_documents(documents)
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[self.reranker]
            )
            self.dirty = True
            tokens = self.count_tokens(functions)
        finally:
            self.lock.writer_release()
            end = time.time()
            print(f"FunctionsManager: push_functions took {end - start} seconds")
            return tokens, {end-start}