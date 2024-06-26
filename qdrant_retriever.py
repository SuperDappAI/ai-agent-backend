# Importing necessary libraries and modules
from datetime import datetime
from enum import Enum
from langchain.schema import BaseRetriever, Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from datetime import timedelta
from langchain_qdrant import Qdrant
from rate_limiter import RateLimiter, SyncRateLimiter
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
    rate_limiter: RateLimiter
    rate_limiter_sync: SyncRateLimiter
    collection_name: str

    client: QdrantClient

    vectorstore: Qdrant
    """The vectorstore to store documents and determine salience."""

    extra_index_penalty: float = float(0.1)

    subconscious_memory_penalty: float = float(0.05)
    """Penalty given to the combined score (percentage) if the memory type is SUBCONSCIOUS_MEMORY."""

    _max_summarizations: int = int(20)
    """How many summaries before we prune unused memories."""

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

    async def get_salient_docs(self, query: str, **kwargs) -> List[Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        return await self.rate_limiter.execute(self.vectorstore.asimilarity_search_with_score, query, k=10, **kwargs)

    async def get_relevant_documents_for_reflection(
        self, query: str, conversation: str, **kwargs
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now()
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.importance",
                    match=rest.MatchValue(value="high"),
                )
            ]
        )
        kwargs.update({"filter": filter})
        docs_and_scores = await self.get_salient_docs(query, **kwargs)
        rescored_docs = []
        for doc, relevance in docs_and_scores:
            combined_score = self._get_combined_score(
                doc, relevance, conversation)
            # Skip the document if it matches the given query, and conversation
            if doc.page_content == query and doc.metadata["extra_index"] == conversation:
                continue  # Skip to the next iteration
            rescored_docs.append((doc, combined_score))
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        # only look at the top 3 results
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
        results, _ = self.rate_limiter_sync.execute(
            self.client.scroll, collection_name=self.collection_name, scroll_filter=filter, limit=5000)
        docs = []
        for record in results:
            document = self.vectorstore._document_from_scored_point(
                record, self.collection_name, self.vectorstore.content_payload_key, self.vectorstore.metadata_payload_key
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

    def _get_relevant_documents(self, *args, **kwargs):
        pass

    async def _aget_relevant_documents(
        self, query: str, **kwargs
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now().timestamp()
        extra_index = kwargs.pop("extra_index", None)
        user_filter = kwargs.pop("user_filter", None)
        if user_filter:
            kwargs.update({"filter": user_filter})
        docs_and_scores = await self.get_salient_docs(query, **kwargs)
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, extra_index))
            for doc, relevance in docs_and_scores
        ]
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs:
            doc.metadata["last_accessed_at"] = current_time
            # Decrement the summarizations count
            if 'summarizations' in doc.metadata and doc.metadata['summarizations'] > 0:
                doc.metadata['summarizations'] -= 1

        # Sort by score and extract just the documents
        sorted_docs = [doc for doc, _ in sorted(
            rescored_docs, key=lambda x: x[1], reverse=True)]
        # Return just the list of Documents
        return sorted_docs

    def get_key_value_document(self, key, value) -> Document:
        """Get the key value from vectordb via scrolling."""
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key=key,
                    match=rest.MatchValue(value=value),
                )
            ]
        )
        record, _ = self.rate_limiter_sync.execute(
            self.client.scroll, collection_name=self.collection_name, scroll_filter=filter, limit=1)
        if record is not None and len(record) > 0:
            return self.vectorstore._document_from_scored_point(
                record[0], self.collection_name, self.vectorstore.content_payload_key, self.vectorstore.metadata_payload_key
            )
        else:
            return None

    def delete_max_summarized(self):
        """Prune points that have been summarized more than _max_summarizations times."""
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.summarized",
                    range=rest.Range(gt=self._max_summarizations),
                )
            ]
        )
        self.rate_limiter_sync.execute(
            self.client.delete, collection_name=self.collection_name, points_selector=filter)
