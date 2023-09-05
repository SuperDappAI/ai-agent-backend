import pytest
from qdrant_retriever import QDrantVectorStoreRetriever, MemoryType
from qdrant_client.http import models as rest
from qdrant_client.http.models import PayloadSchemaType
from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from datetime import datetime
from dotenv import load_dotenv
import os

@pytest.fixture
def setup_retriever():
    load_dotenv()
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    client = QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY)  
    collection_name = "test_collection"
    try:
        client.create_collection(
            on_disk_payload=True,
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=1536,
                distance=rest.Distance.COSINE,
            ),
        )
        client.create_payload_index(collection_name, "metadata.extra_index", field_schema=PayloadSchemaType.KEYWORD)
    except:
        print("MemorySummarizer: loaded from cloud...")
    finally:
        vectorstore = Qdrant(client, collection_name, OpenAIEmbeddings())
        nowStamp = datetime.now() 
        metadata = {
            "id": 99999999998,
            "extra_index": "test",
            "created_at": nowStamp,
            "importance": "high", 
            "last_accessed_at": nowStamp,
            "summarizations": 0,
            "group_id": "test_user",
            "memory_type": "summary",
        }
        document = Document(
            page_content="test content lorem ipsum test test test", 
            metadata=metadata,
        )
        
    vectorstore.add_documents([document], ids=[metadata["id"]], wait = False)
    retriever = QDrantVectorStoreRetriever(client=client, vectorstore=vectorstore, collection_name=collection_name)
    return retriever

# def test_get_salient_docs(setup_retriever):
#     retriever = setup_retriever
#     query = "test_query"
#     docs = retriever.get_salient_docs(query)
#     assert isinstance(docs, list)
#     for doc, score in docs:
#         assert isinstance(doc, Document)
#         assert isinstance(score, float)
#         print(doc)

def test_get_salient_docs(setup_retriever):
    retriever = setup_retriever
    query = "test_query"
    docs = retriever.get_salient_docs(query)
    
    assert isinstance(docs, list)
    
    assert len(docs) > 0

    for doc, score in docs:
        assert isinstance(doc, Document)
        
        assert hasattr(doc, 'page_content')
        assert hasattr(doc, 'metadata')

        assert isinstance(score, float)
        
        assert 0.0 <= score <= 1.0