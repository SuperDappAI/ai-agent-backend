# Importing necessary libraries and modules
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pydantic import Field
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.base import VectorStore

def _get_hours_passed(time: datetime, ref_time: datetime) -> float:
    """Get the hours passed between two datetime objects."""
    return (time - ref_time).total_seconds() / 3600


class TimeWeightedVectorStoreRetriever(BaseRetriever):
    """Retriever that combines embedding similarity with recency in retrieving values."""

    vectorstore: VectorStore
    """The vectorstore to store documents and determine salience."""

    search_kwargs: dict = Field(default_factory=lambda: dict(k=100))
    """Keyword arguments to pass to the vectorstore similarity search."""

    decay_rate: float = Field(default=0.01)
    """The exponential decay factor used as (1.0-decay_rate)**(hrs_passed)."""

    conversation_bonus: float = Field(default=0.1)
    """Bonus given to semantic scoring (percentage) if we are searching across conversations and it is in the conversation that is doing reflection."""

    other_score_keys: List[str] = []
    """Other keys in the metadata to factor into the score, e.g. 'importance'."""

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: Optional[float],
        current_time: datetime,
        conversation: str = None
    ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            document.metadata["last_accessed_at"],
        )
        score = (1.0 - self.decay_rate) ** hours_passed
        for key in self.other_score_keys:
            if key in document.metadata:
                score += document.metadata[key]
        if vector_relevance is not None:
            score += vector_relevance
        if conversation is not None and conversation == document.metadata.get("conversation"):
            score += self.conversation_bonus
        return score

    def get_salient_docs(self, query: str) -> List[Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        return self.vectorstore.similarity_search_with_relevance_scores(query, **self.search_kwargs)

    def get_relevant_documents_for_reflection(
        self, query: str, conversation: str
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now()
        oldargs = self.search_kwargs.copy()
        self.search_kwargs.update({"filter": {"importance": 9, "importance": 10}, "k": 10})
        docs_and_scores = self.get_salient_docs(query)
        self.search_kwargs = oldargs
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time, conversation))
            for doc, relevance in docs_and_scores
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        # only look at the top 3 results out of 10
        if len(rescored_docs) > 3:
            rescored_docs = rescored_docs[-3:]
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs:
            doc.metadata["last_accessed_at"] = current_time
        # Return just the list of Documents
        return [doc for doc, _ in rescored_docs]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now()
        docs_and_scores = self.get_salient_docs(query)
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs:
            doc.metadata["last_accessed_at"] = current_time
        # Return just the list of Documents
        return [doc for doc, _ in rescored_docs]

