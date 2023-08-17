import unittest
from unittest.mock import patch, MagicMock
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from memory_summarizer import MemorySummarizer, FlexibleDocumentSummarizer, FlexibleDocumentTreeSummarizer  

# TODO: revise, add pytest
class TestMemorySummarizer(unittest.TestCase):

    def setUp(self):
        self.agentManagerMock = MagicMock()
        self.summarizer = MemorySummarizer(self.agentManagerMock)
        self.documents_mock = [MagicMock(metadata={"group_id": 1, "extra_index": 1, "created_at": x}) for x in range(10)]

    @patch('os.getenv')
    @patch('memory_summarizer.AsyncIOScheduler')
    @patch('memory_summarizer.FlexibleDocumentSummarizer')
    @patch('memory_summarizer.FlexibleDocumentTreeSummarizer')
    # def test_init(self, *args):
    #     summarizer = MemorySummarizer(self.agentManagerMock)
    #     self.assertIsInstance(summarizer.flexible_document_summarizer, FlexibleDocumentSummarizer)
    #     self.assertIsInstance(summarizer.flexible_document_tree_summarizer, FlexibleDocumentTreeSummarizer)
    #     self.assertIsInstance(summarizer.scheduler, AsyncIOScheduler)
    #     self.assertFalse(summarizer.summarizing)
    
    def test_sort_and_group_documents(self):
        groups = self.summarizer._sort_and_group_documents(self.documents_mock)
        self.assertEqual(len(groups), 1) 
        self.assertEqual(len(groups[0]), 10) 
        for idx, doc in enumerate(sorted(self.documents_mock, key=lambda x: x.metadata["created_at"])):
            self.assertEqual(doc, groups[0][idx])

    @patch.object(BaseScheduler, 'start')
    def test_start(self, start_mock):
        self.summarizer.start()
        start_mock.assert_called_once()

    @patch.object(BaseScheduler, 'shutdown')
    def test_stop(self, shutdown_mock):
        self.summarizer.stop()
        shutdown_mock.assert_called_once()

if __name__ == "__main__":
    unittest.main()
