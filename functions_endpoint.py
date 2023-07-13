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

load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("PINECONE_API_KEY")

pinecone.init()

#load json from file

class FunctionsManager:

    def __init__(self):
        user_id = "functions_test"
        self.user_id = user_id

    def transform(self,data,category):
        result = []
        for item in data[category]:
            entry = str(item)
            hash = hashlib.sha256(entry.encode())
            page_content = f"{item['name']}. {item['description']}. {category}"
            metadata = {'name': item['name'], 'category': category, 'hash': hash.hexdigest()}
            result.append({'page-content': page_content, 'metadata': metadata})
        return result

    def transform_and_push(self,data):

        formatted = []
        category = 'informationretrieval_functions'

        #manually loading just info and comm functions

        informationretrieval_functons = self.transform(data, 'informationretrieval_functions')
        communication_functions = self.transform(data, 'communication_functions')
        dataprocessing_functions = self.transform(data, 'dataprocessing_functions')
        # sensory perception
        sensory_perception = self.transform(data, 'sensoryperception_functions')
        try:
            # memory
            memory_functions = self.transform(data, 'memory_functions')
            # decision making
            decision_making = self.transform(data, 'decisionmaking_functions')
            # learning
            learning_functions = self.transform(data, 'learning_functions')
        except:
            print("Not implemented yet")

        #push to pinecone
        info_docs = []
        for doc in informationretrieval_functons:
            info_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))

        comm_docs = []
        for doc in communication_functions:
            comm_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))

        dataprocessing_docs = []
        for doc in dataprocessing_functions:
            dataprocessing_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))

        sensory_perception_docs = []
        for doc in sensory_perception:
            sensory_perception_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))

        try:
            memory_docs = []
            for doc in memory_functions:
                memory_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))
            
            decision_making_docs = []
            for doc in decision_making:
                decision_making_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))

            learning_docs = []
            for doc in learning_functions:
                learning_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))
        except:
            print("Not implemented yet")

        pinecone_db = Pinecone.from_existing_index(
                    "aida", embedding=OpenAIEmbeddings(), namespace="functions_test"
                )
        #count operation time

        native_index_object = pinecone.Index("aida")
        native_index_object.delete(namespace="functions_test", delete_all=True)

        print("Deleted all functions from index")

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
            print("Error adding memory, decision making or learning functions")

        print("Added all functions to index")

        end = time.time()

        print(f"Operation took {end - start} seconds")
        return info_docs,comm_docs,dataprocessing_docs

