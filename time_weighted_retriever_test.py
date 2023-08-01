import unittest
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from time_weighted_retriever import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import VectorStore
from datetime import datetime, timedelta

class TestTimeWeightedVectorStoreRetriever(unittest.TestCase):
    def setUp(self):
        self.vector_store = MagicMock(spec=VectorStore)
        self.retriever = TimeWeightedVectorStoreRetriever(vectorstore=self.vector_store)
        self.doc1 = Document(page_content='Hello, world!', metadata={'last_accessed_at': datetime.now() - timedelta(hours=2), 'importance': 5})
        self.doc2 = Document(page_content='Goodbye, world!', metadata={'last_accessed_at': datetime.now() - timedelta(hours=1), 'importance': 3})

    def test_get_combined_score(self):
        vector_relevance = 0.5
        current_time = datetime.now()
        conversation = "test_conversation"
        score = self.retriever._get_combined_score(self.doc1, vector_relevance, current_time, conversation)
        self.assertIsInstance(score, float, f"Expected score of type float, but got type {type(score)}")

    def test_get_relevant_documents_for_reflection(self):
        query = "test_query"
        conversation = "test_conversation"
        self.vector_store.similarity_search_with_relevance_scores.return_value = [(self.doc1, 0.75), (self.doc2, 0.5)]
        docs = self.retriever.get_relevant_documents_for_reflection(query, conversation)
        self.assertEqual(docs, [self.doc1, self.doc2], f"Expected docs [doc1, doc2], but got {docs}")
        self.vector_store.similarity_search_with_relevance_scores.assert_called_with(query, k=10, filter={'importance_score': 10})

    def test_get_salient_docs(self):
        query = "test_query"
        self.vector_store.similarity_search_with_relevance_scores.return_value = [(self.doc1, 0.75), (self.doc2, 0.5)]
        docs = self.retriever.get_salient_docs(query)
        self.assertEqual(docs, [(self.doc1, 0.75), (self.doc2, 0.5)])
        self.vector_store.similarity_search_with_relevance_scores.assert_called_with(query, k=100)

    def test_get_relevant_documents(self):
        query = "test_query"
        self.vector_store.similarity_search_with_relevance_scores.return_value = [(self.doc1, 0.75), (self.doc2, 0.5)]
        docs = self.retriever.get_relevant_documents(query)
        self.assertEqual(docs, [self.doc1, self.doc2])
        self.vector_store.similarity_search_with_relevance_scores.assert_called_with(query, k=100)

if __name__ == "__main__":
    unittest.main()
