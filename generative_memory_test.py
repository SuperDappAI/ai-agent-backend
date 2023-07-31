import unittest
from unittest.mock import MagicMock, patch
from your_module_path import Document, GenerativeAgentMemory, TimeWeightedVectorStoreRetriever

class TestGenerativeAgentMemory(unittest.TestCase):
    def setUp(self):
        self.time_weighted_retriever = MagicMock(spec=TimeWeightedVectorStoreRetriever)
        self.memory = GenerativeAgentMemory(retriever=self.time_weighted_retriever)
        self.doc1 = Document(id='1', text='Hello, world!', vector=[0.1, 0.2, 0.3], metadata={})
        self.doc2 = Document(id='2', text='Goodbye, world!', vector=[0.4, 0.5, 0.6], metadata={})

    def test_add_and_save(self):
        self.memory.add_and_save(self.doc1)
        self.memory.retriever.vector_store.add.assert_called_once_with(self.doc1)
        self.memory.retriever.vector_store.save.assert_called_once()

    def test_retrieve(self):
        query = "test query"
        conversation = "test conversation"
        self.memory.retriever.get_salient_docs.return_value = [(self.doc1, 0.75), (self.doc2, 0.5)]
        docs = self.memory.retrieve(query, conversation)
        self.assertEqual(docs, [(self.doc1, 0.75), (self.doc2, 0.5)])
        self.memory.retriever.get_salient_docs.assert_called_once_with(query, conversation)

    def test_index_memory(self):
        conversation = "test_conversation"
        reflections = "test_reflections"
        salient_doc_ids = ["1", "2"]
        self.memory.index_memory(conversation, reflections, salient_doc_ids)
        self.memory.retriever.index.assert_called_once_with(conversation, reflections, salient_doc_ids)

    def test_get_memory_of_conversation(self):
        conversation = "test_conversation"
        self.memory.get_memory_of_conversation(conversation)
        self.memory.retriever.get_documents_of_conversation.assert_called_once_with(conversation)

    def test_search(self):
        query = "test query"
        num_results = 5
        self.memory.retriever.vector_store.search.return_value = [self.doc1, self.doc2]
        docs = self.memory.search(query, num_results)
        self.assertEqual(docs, [self.doc1, self.doc2])
        self.memory.retriever.vector_store.search.assert_called_once_with(query, num_results)

if __name__ == "__main__":
    unittest.main()
