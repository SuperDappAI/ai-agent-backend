import pinecone
import time
from dotenv import load_dotenv
from llama_index import OpenAI, ServiceContext, LLMRerank, Document, VectorStoreIndex, StorageContext, load_index_from_storage
import os
import tiktoken
import schedule
import threading
from pathlib import Path

load_dotenv()
os.getenv("OPENAI_API_KEY")

pinecone.init()

#load json from file

class FunctionsManager1:
    def __init__(self):
        self.dirpath = Path("./storage_functions")
        self.index = None
        self.query_engine = None
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, service_context=ServiceContext.from_defaults(
            llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
        ))
        self.load()
        schedule.every(300).to(600).seconds.do(self.save)
        thread = threading.Thread(target=self.run_continuously)
        thread.start()
        
    def run_continuously(self):
        while True:
            schedule.run_pending()
            time.sleep(1)

    def transform(self,data,category):
        result = []
        for item in data[category]:
            page_content = {'name': item['name'], 'description': str(item['description']), 'category': category}
            result.append(page_content)
        return result

    def save(self):
        start = time.time()
        self.saving = True
        for doc in self.index:
            if doc.dirty is True:
                doc.storage_context.persist(persist_dir=self.dirpath)
                doc.dirty = False
        self.saving = False
        end = time.time()
        print(f"FunctionsManager: Save operation took {end - start} seconds")

    def count_tokens(self, functions):
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
        if self.query_engine is None:
            print("FunctionsManager: Error pull_functions, query_engine doesn't exist")
            return None
        response = self.query_engine.query(
            query, 
        )
        return response

    def load(self):
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

    def push_functions(self, functions):
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



