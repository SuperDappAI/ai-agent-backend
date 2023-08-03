import pinecone
import logging
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

# load json from file

with open('utils/functions_one_by_one.json') as f:
    data = json.load(f)


def transform(data, category):
    result = []
    for item in data[category]:
        entry = str(item)
        hash = hashlib.sha256(entry.encode())
        page_content = f"{str(item)}.Category: {category}."
        metadata = {'name': item['Function'],
                    'category': category, 'hash': hash.hexdigest()}
        result.append({'page-content': page_content, 'metadata': metadata})
    return result


formatted = []
category = 'Information Retrieval'

# manually loading just info and comm functions

informationretrieval_functons = transform(data, 'Information Retrieval')
communication = transform(data, 'Communication')
data_processing = transform(data, 'Data Processing')
sensory_perception = transform(data, 'Sensory Perception')

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

sensoryperception_docs = []
for doc in sensory_perception:
    sensoryperception_docs.append(
        Document(page_content=doc['page-content'], metadata=doc['metadata']))

pinecone_db = Pinecone.from_existing_index(
    "aida", embedding=OpenAIEmbeddings(), namespace="functions_test"
)

# count operation time
start = time.time()
pinecone_db.as_retriever().add_documents(info_docs)
pinecone_db.as_retriever().add_documents(comm_docs)
pinecone_db.as_retriever().add_documents(dataprocessing_docs)
pinecone_db.as_retriever().add_documents(sensoryperception_docs)
end = time.time()

logging.info(f"Operation took {end - start} seconds")
