import pinecone
import time
from langchain.schema import Document
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import json
import hashlib
import json
import tiktoken


load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("PINECONE_API_KEY")

pinecone.init()

# load json from file


class FunctionsManager:

    def __init__(self):
        user_id = "functions_test"
        self.user_id = user_id

    def transform(self, data, category, mode):
        result = []
        for item in data[category]:
            entry = str(item)
            hash = hashlib.sha256(entry.encode())
            if mode == 0:
                page_content = f"{str(item['name'])}: {str(item['description'])}"
            if mode == 1:
                page_content = f"{str(item)}"
            if mode == 2:
                page_content = f"{str(item['name'])}: {str(item['examples'])}"
            try:
                metadata = {
                    'name': item['name'], 'category': category, 'hash': hash.hexdigest()}
            except:
                metadata = {
                    'name': item['Function'], 'category': category, 'hash': hash.hexdigest()}
            result.append({'page-content': page_content, 'metadata': metadata})
        return result

    def count_tokens(self, functions):
        mode = 1
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        informationretrieval_functons = self.transform(
            functions, 'information_retrieval', mode)
        communication = self.transform(functions, 'communication', mode)
        data_processing = self.transform(functions, 'data_processing', mode)
        # sensory perception
        sensory_perception = self.transform(
            functions, 'sensory_perception', mode)

        info_functions = []
        for doc in informationretrieval_functons:
            info_functions.append(
                Document(page_content=doc['page-content'], metadata=doc['metadata']))

        comm_functions = []
        for doc in communication:
            comm_functions.append(
                Document(page_content=doc['page-content'], metadata=doc['metadata']))

        data_processing = []
        for doc in data_processing:
            data_processing.append(
                Document(page_content=doc['page-content'], metadata=doc['metadata']))

        sensory_perception_functions = []
        for doc in sensory_perception:
            sensory_perception_functions.append(
                Document(page_content=doc['page-content'], metadata=doc['metadata']))

        all_docs = info_functions + comm_functions + \
            data_processing + sensory_perception_functions
        tokens = []
        for doc in all_docs:
            tokens.append({doc.metadata['name']: len(
                encoding.encode(doc.page_content))})
        return tokens

    def transform_and_push(self, functions, examples, namespace, mode):

        formatted = []
        category = 'information_retrieval'

        # manually loading just info and comm functions

        informationretrieval_functons = self.transform(
            examples, 'Information Retrieval', mode)
        communication = self.transform(examples, 'Communication', mode)
        data_processing = self.transform(examples, 'Data Processing', mode)
        # sensory perception
        sensory_perception = self.transform(
            examples, 'Sensory Perception', mode)
        try:
            # memory
            memory_functions = self.transform(examples, 'Memory', mode)
            # decision making
            decision_making = self.transform(examples, 'Decision', mode)
            # learning
            learning_functions = self.transform(examples, 'Learning', mode)
        except:
            logging.info("Not implemented yet")

        # push to pinecone
        info_docs = []
        for doc in informationretrieval_functons:
            info_docs.append(
                Document(page_content=doc['page-content'], metadata=doc['metadata']))

        comm_docs = []
        for doc in communication:
            comm_docs.append(
                Document(page_content=doc['page-content'], metadata=doc['metadata']))

        dataprocessing_docs = []
        for doc in data_processing:
            dataprocessing_docs.append(
                Document(page_content=doc['page-content'], metadata=doc['metadata']))

        sensory_perception_docs = []
        for doc in sensory_perception:
            sensory_perception_docs.append(
                Document(page_content=doc['page-content'], metadata=doc['metadata']))

        try:
            memory_docs = []
            for doc in memory_functions:
                memory_docs.append(
                    Document(page_content=doc['page-content'], metadata=doc['metadata']))

            decision_making_docs = []
            for doc in decision_making:
                decision_making_docs.append(
                    Document(page_content=doc['page-content'], metadata=doc['metadata']))

            learning_docs = []
            for doc in learning_functions:
                learning_docs.append(
                    Document(page_content=doc['page-content'], metadata=doc['metadata']))
        except:
            logging.info("Not implemented yet")

        pinecone_db = Pinecone.from_existing_index(
            "aida", embedding=OpenAIEmbeddings(), namespace=namespace
        )
        # count operation time

        native_index_object = pinecone.Index("aida")
        native_index_object.delete(namespace=namespace, delete_all=True)

        logging.info("Deleted all functions from index")

        start = time.time()
        pinecone_db.as_retriever().add_documents(info_docs)
        pinecone_db.as_retriever().add_documents(comm_docs)
        pinecone_db.as_retriever().add_documents(dataprocessing_docs)
        pinecone_db.as_retriever().add_documents(sensory_perception_docs)
        try:
            pinecone_db.as_retriever().add_documents(memory_docs)
            pinecone_db.as_retriever().add_documents(decision_making_docs)
            pinecone_db.as_retriever().add_documents(learning_docs)
        except:
            logging.info(
                "Error adding memory, decision making or learning functions")

        logging.info("Added all functions to index")

        end = time.time()

        logging.info(f"Operation took {end - start} seconds")

        tokens = self.count_tokens(functions)

        return tokens
