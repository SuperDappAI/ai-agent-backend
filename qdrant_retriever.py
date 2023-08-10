# Importing necessary libraries and modules
from datetime import datetime
from pydantic import Field
from enum import Enum
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from datetime import timedelta
from langchain.vectorstores import Qdrant
from typing import (
    List,
    Optional,
    Tuple,
)

class MemoryType(Enum):
    CONSCIOUS_MEMORY = 0
    SUBCONSCIOUS_MEMORY = 1
    
class QDrantVectorStoreRetriever(BaseRetriever):
    """Retriever that combines embedding similarity with conversation matching scores in retrieving values."""
    collection_name: str
    
    client: QdrantClient
    
    vectorstore: Qdrant
    """The vectorstore to store documents and determine salience."""

    extra_index_penalty: float = Field(default=0.1)
    
    subconscious_memory_penalty: float = Field(default=0.05)
    """Penalty given to the combined score (percentage) if the memory type is SUBCONSCIOUS_MEMORY."""

    _max_summarizations: int = int(2)
    """How many summaries before we tree summaries and prune unused memories."""
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: Optional[float],
        extra_index: str = None
    ) -> float:
        """Return the combined score for a document."""
        score = 0
        if vector_relevance is not None:
            score += vector_relevance
        if extra_index is not None and extra_index != document.metadata.get("extra_index"):
            score -= self.extra_index_penalty
        if document.metadata.get("memory_type") == MemoryType.SUBCONSCIOUS_MEMORY:
            score -= self.subconscious_memory_penalty
        return score

    def get_salient_docs(self, query: str, **kwargs) -> List[Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        return self.vectorstore.similarity_search_with_score(query, k=10, **kwargs)

    def get_relevant_documents_for_reflection(
        self, query: str, user_id: str, conversation: str, **kwargs
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now()
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.group_id", 
                    match=rest.MatchValue(value=user_id), 
                ),
                rest.FieldCondition(
                    key="metadata.importance", 
                    match=rest.MatchValue(value="high"), 
                )
            ]
        )
        kwargs.update({"filter": filter})
        docs_and_scores = self.get_salient_docs(query, **kwargs)
        rescored_docs = []
        for doc, relevance in docs_and_scores:
            combined_score = self._get_combined_score(doc, relevance, conversation)
            # Skip the document if it matches the given query, user_id, and conversation
            if doc.page_content == query and doc.metadata["group_id"] == user_id and doc.metadata["extra_index"] == conversation:
                continue  # Skip to the next iteration
            rescored_docs.append((doc, combined_score))
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        # only look at the top 3 results out of 10
        if len(rescored_docs) > 3:
            rescored_docs = rescored_docs[:3]
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs:
            doc.metadata["last_accessed_at"] = current_time.timestamp()
        # Return just the list of Documents
        return [doc for doc, _ in rescored_docs]
    
    def get_documents_for_summarization(self) -> List[Document]:
        """Return documents that are relevant to summarize."""
        current_time = datetime.now()
        two_weeks_ago = current_time - timedelta(weeks=2)
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.last_accessed_at", 
                    range=rest.Range(lte=two_weeks_ago.timestamp()), 
                ),
                rest.FieldCondition(
                    key="metadata.summarizations", 
                    range=rest.Range(lt=self._max_summarizations), 
                )
            ]
        )
        results, _ = self.client.scroll(collection_name=self.collection_name, scroll_filter=filter, limit = 5000)
        docs = []
        for record in results:
            document = self.vectorstore._document_from_scored_point(
                record, self.vectorstore.content_payload_key, self.vectorstore.metadata_payload_key
            )

            # Increment the summarizations count
            if 'summarizations' in document.metadata:
                document.metadata['summarizations'] += 1
            else:
                document.metadata['summarizations'] = 1
            # summarize once every 2 weeks
            document.metadata["last_accessed_at"] = current_time.timestamp()
            docs.append(document)
        return docs

    def get_documents_for_tree_summarization(self) -> List[Document]:
        """Return documents that are relevant to summarize."""
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.summarizations", 
                    range=rest.Range(gte=self._max_summarizations, lt=100), 
                )
            ]
        )
        results, _ = self.client.scroll(collection_name=self.collection_name, scroll_filter=filter, limit = 5000)
        docs = []
        for record in results:
            document = self.vectorstore._document_from_scored_point(
                record, self.vectorstore.content_payload_key, self.vectorstore.metadata_payload_key
            )

            docs.append(document)

        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now().timestamp()
        extra_index = kwargs.pop("extra_index", None)
        docs_and_scores = self.get_salient_docs(query, **kwargs)
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, extra_index))
            for doc, relevance in docs_and_scores
        ]
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs:
            doc.metadata["last_accessed_at"] = current_time
            if 'summarizations' in doc.metadata:
                if doc.metadata['summarizations'] == 100:
                    doc.metadata['summarizations'] = 0
        # Sort by score and extract just the documents
        sorted_docs = [doc for doc, _ in sorted(rescored_docs, key=lambda x: x[1], reverse=True)]
        # Return just the list of Documents
        return sorted_docs

    def clear_using_extra_index(self, extra_index) -> None:
        """Clear memory contents."""
        filter_dict = {
            'must': [
                {
                    'key': 'metadata.extra_index',
                    'match': {'value': extra_index}
                }
            ]
        }
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.extra_index", 
                    match=rest.MatchValue(value=extra_index), 
                )
            ]
        )
        self.client.delete(collection_name=self.collection_name, points_selector=filter, wait = False)

    def does_key_exist(self, key, value):
        """Does the hash of the web content exist in our cache?."""
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key=key, 
                    match=rest.MatchValue(value=value), 
                )
            ]
        )
        results = self.client.scroll(collection_name=self.collection_name, scroll_filter=filter, limit = 1)
        return results is not None and len(results[0]) > 0

    def prune_from(self, fromTime: float):
        """Prune points that are older than fromTime timestamp."""
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.last_accessed_at", 
                    match=rest.Range(lte=fromTime), 
                )
            ]
        )
        self.client.delete(collection_name=self.collection_name, points_selector=filter, wait = False)

    def delete_documents(self, documents: List[Document]) -> None:
        """Delete the documents that have been summarized from the given Qdrant collection."""
        
        # Extract IDs of the documents to be deleted
        points_to_delete = [doc.metadata["id"] for doc in documents]
        
        # Use the Qdrant client to delete the documents by their IDs
        self.client.delete(collection_name=self.collection_name,
                        points_selector=rest.PointIdsList(points=points_to_delete), wait=True)

