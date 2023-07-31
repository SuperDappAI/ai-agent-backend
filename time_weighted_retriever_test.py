import unittest
from unittest.mock import MagicMock, patch
from your_module_path import Document, TimeWeightedVectorStoreRetriever, VectorStore
from datetime import datetime, timedelta

class TestTimeWeightedVectorStoreRetriever(unittest.TestCase):
    def setUp(self):
        self.vector_store = MagicMock(spec=VectorStore)
        self.retriever = TimeWeightedVectorStoreRetriever(vectorstore=self.vector_store)
        self.doc1 = Document(id='1', text='Hello, world!', vector=[0.1, 0.2, 0.3], metadata={'last_accessed_at': datetime.now() - timedelta(hours=2), 'importance': 5})
        self.doc2 = Document(id='2', text='Goodbye, world!', vector=[0.4, 0.5, 0.6], metadata={'last_accessed_at': datetime.now() - timedelta(hours=1), 'importance': 3})

    def test_get_combined_score(self):
        vector_relevance = 0.5
        current_time = datetime.now()
        conversation = "test_conversation"
        score = self.retriever._get_combined_score(self.doc1, vector_relevance, current_time, conversation)
        self.assertIsInstance(score, float)

    def test_get_salient_docs(self):
        query = "test_query"
        conversation = "test_conversation"
        self.vector_store.similarity_search_with_relevance_scores.return_value = [(self.doc1, 0.75), (self.doc2, 0.5)]
        docs = self.retriever.get_salient_docs(query, conversation)
        self.assertEqual(docs, [(self.doc1, 0.75), (self.doc2, 0.5)])
        self.vector_store.similarity_search_with_relevance_scores.assert_called_once_with(query)

    def test_get_relevant_documents_for_reflection(self):
        query = "test_query"
        conversation = "test_conversation"
        self.vector_store.similarity_search_with_relevance_scores.return_value = [(self.doc1, 0.75), (self.doc2, 0.5)]
        docs = self.retriever.get_relevant_documents_for_reflection(query, conversation)
        self.assertEqual(docs, [self.doc1, self.doc2])
        self.vector_store.similarity_search_with_relevance_scores.assert_called_once_with(query)

    def test_get_relevant_documents(self):
        query = "test_query"
        self.vector_store.similarity_search_with_relevance_scores.return_value = [(self.doc1, 0.75), (self.doc2, 0.5)]
        docs = self.retriever._get_relevant_documents(query)
        self.assertEqual(docs, [self.doc1, self.doc2])
        self.vector_store.similarity_search_with_relevance_scores.assert_called_once_with(query)

if __name__ == "__main__":
    unittest.main()
