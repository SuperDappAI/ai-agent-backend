import pinecone
import time
from langchain.schema import Document
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import json
import hashlib

load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("PINECONE_API_KEY")

pinecone.init()

#load json from file

import json
with open('utils/functions3.json') as f:
    data = json.load(f)

def transform(data, category):
    result = []
    for item in data[category]:
        entry = str(item)
        hash = hashlib.sha256(entry.encode())
        page_content = f"{str(item)}"
        metadata = {'name': item['name'], 'category': category, 'hash': hash.hexdigest()}
        result.append({'page-content': page_content, 'metadata': metadata})
    return result

formatted = []
category = 'informationretrieval_functions'

#manually loading just info and comm functions

informationretrieval_functons = transform(data, 'informationretrieval_functions')
communication_functions = transform(data, 'communication_functions')
dataprocessing_functions = transform (data, 'dataprocessing_functions')
sensoryperception_functions = transform(data, 'sensoryperception_functions')

info_docs = []
for doc in informationretrieval_functons:
    info_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))

comm_docs = []
for doc in communication_functions:
    comm_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))

dataprocessing_docs = []
for doc in dataprocessing_functions:
    dataprocessing_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))

sensoryperception_docs = []
for doc in sensoryperception_functions:
    sensoryperception_docs.append(Document(page_content=doc['page-content'],metadata=doc['metadata']))

pinecone_db = Pinecone.from_existing_index(
            "aida", embedding=OpenAIEmbeddings(), namespace="functions_test2"
        )

#count operation time
start = time.time()
pinecone_db.as_retriever().add_documents(info_docs)
pinecone_db.as_retriever().add_documents(comm_docs)
pinecone_db.as_retriever().add_documents(dataprocessing_docs)
pinecone_db.as_retriever().add_documents(sensoryperception_docs)
end = time.time()

print(f"Operation took {end - start} seconds")

