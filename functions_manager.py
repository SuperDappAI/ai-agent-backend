import time
from dotenv import load_dotenv
from llama_index import OpenAI, ServiceContext, LLMRerank, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from reader_writer_lock import ReaderWriterLock
import os
import tiktoken
import schedule
import threading
from pathlib import Path
import json

class FunctionsManager1:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        self.dirpath = Path("./storage_functions")
        self.index = None
        self.query_engine = None
        self.lock = ReaderWriterLock()
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, 
            service_context=ServiceContext.from_defaults(
                llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
            ))
        # Try to load index data from the filesystem
        if not self.load():
            # If loading was unsuccessful (e.g., no data on the filesystem), load functions from JSON file
            with open('./utils/functions.json', 'r') as f:
                functions_json = json.load(f)
            self.push_functions(functions_json)

        # Save function scheduled to run every 5 to 10 minutes
        schedule.every(300).to(600).seconds.do(self.save)
        
        # Create new thread for schedule
        threading.Thread(target=self.run_continuously).start()

    def run_continuously(self):
        """Keep checking and running pending tasks every second."""
        while True:
            schedule.run_pending()
            time.sleep(1)

    def transform(self, data, category):
        """Transforms function data for a specific category."""
        result = []
        for item in data[category]:
            page_content = {'name': item['name'], 'description': str(item['description']), 'category': category}
            result.append(page_content)
        return result

    def save(self):
        """Persist current index data to the filesystem."""
        self.lock.writer_acquire()
        try:
            start = time.time()
            for doc in self.index:
                if doc.dirty is True:
                    doc.storage_context.persist(persist_dir=self.dirpath)
                    doc.dirty = False
            end = time.time()
            print(f"FunctionsManager: Save operation took {end - start} seconds")
        finally:
            self.lock.writer_release()

    def count_tokens(self, functions):
        """Count the tokens for all the functions."""
        function_types = ['informationretrieval_functions', 
                        'communication_functions', 
                        'dataprocessing_functions', 
                        'sensoryperception_functions']

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = [] 
        for func_type in function_types:
            for doc in functions[func_type]:
                tokens.append({doc['name']: len(encoding.encode(doc))})
        return tokens

    def pull_functions(self, query):
        """Fetch functions based on a query."""
        self.lock.reader_acquire()
        try:
            if self.query_engine is None:
                print("FunctionsManager: Error pull_functions, query_engine doesn't exist")
                return None
            response = self.query_engine.query(query)
            return response
        finally:
            self.lock.reader_release()

    def load(self):
        """Load existing index data from the filesystem."""
        self.lock.writer_acquire()
        try:
            print("FunctionsManager: Loading from disk")
            start = time.time()
            if self.dirpath.exists() and self.dirpath.is_dir():
                # rebuild storage context
                storage_context = StorageContext.from_defaults(persist_dir=self.dirpath)

                # load index
                self.index = load_index_from_storage(storage_context)
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=10,
                    node_postprocessors=[self.reranker]
                )
                end = time.time()
                print(f"FunctionsManager: Load took {end - start} seconds")
                return True  # Return True when loading is successful
            else:
                return False  # Return False when there's no data to load
        finally:
            self.lock.writer_release()

    def push_functions(self, functions):
        """Update the current index with new functions."""
        self.lock.writer_acquire()
        try:
            print("FunctionsManager: Deleting persistent directory, adding functions to index and persisting to disk...")
            start = time.time()

            function_types = ['informationretrieval_functions', 
                            'communication_functions', 
                            'dataprocessing_functions', 
                            'sensoryperception_functions']

            all_docs = []

            # Transform and concatenate function types
            for func_type in function_types:
                transformed_functions = self.transform(functions[func_type], func_type.replace('_', ' ').title())
                all_docs.extend(transformed_functions)
            
            documents = [Document(t) for t in all_docs]
        
            self.index = VectorStoreIndex.from_documents(documents)
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[self.reranker]
            )
            self.index.dirty = True

            end = time.time()

            print(f"FunctionsManager: push_functions took {end - start} seconds")
            tokens = self.count_tokens(functions)

            return tokens
        finally:
            self.lock.writer_release()

