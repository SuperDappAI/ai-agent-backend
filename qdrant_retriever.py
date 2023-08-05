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
    Any,
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

    def _build_condition(self, key: str, value: Any) -> List[rest.FieldCondition]:
        out = []

        if 'match' in value:
            condition_args = {
                "key": key,
                "match": rest.MatchValue(value=value['match']['value'])
            }
            out.append(rest.FieldCondition(**condition_args))
        elif 'range' in value:
            condition_args = {
                "key": key,
                "range": rest.Range(**value['range'])
            }
            out.append(rest.FieldCondition(**condition_args))

        return out

    def _qdrant_filter_from_dict(self, filter: Optional[Any]) -> Optional[rest.Filter]:
        if not filter:
            return None

        # Separate conditions based on must, should, and must_not
        must_conditions = []
        should_conditions = []
        must_not_conditions = []

        # Check if the filter is a list or dictionary, and handle accordingly
        if isinstance(filter, list):
            for item in filter:
                key = item['key']
                value = item['value']
                must_conditions.extend(self._build_condition(key, value))
        else:
            for key, value in filter.items():
                if key == 'must':
                    must_conditions.extend(self._build_condition("", value))
                elif key == 'should':
                    should_conditions.extend(self._build_condition("", value))
                elif key == 'must_not':
                    must_not_conditions.extend(self._build_condition("", value))
                else:
                    must_conditions.extend(self._build_condition(key, value))  # Default to must if not specified

        return rest.Filter(
            must=must_conditions if must_conditions else None,
            should=should_conditions if should_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None
        )

    def get_salient_docs(self, query: str, **kwargs) -> List[Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        return self.vectorstore.similarity_search_with_relevance_scores(query, **kwargs)

    def get_relevant_documents_for_reflection(
        self, query: str, user_id: str, conversation: str, **kwargs
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now()
        filter_dict = {
            'must': [
                {
                    'key': 'metadata.group_id',
                    'match': {'value': user_id}
                },
                {
                    'key': 'metadata.importance_score',
                    'range': {'gte': 8} # gte stands for 'greater than or equal to'
                }
            ]
        }
        filter = self._qdrant_filter_from_dict(filter_dict)
        kwargs.update({"filter": filter})
        docs_and_scores = self.get_salient_docs(query, **kwargs)
        min_score = kwargs.get("score_threshold")
        rescored_docs = []
        for doc, relevance in docs_and_scores:
            combined_score = self._get_combined_score(doc, relevance, conversation)
            # Skip the document if it matches the given query, user_id, and conversation
            if doc.page_content == query and doc.metadata["group_id"] == user_id and doc.metadata["extra_index"] == conversation:
                continue  # Skip to the next iteration
            if combined_score >= min_score:
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
    
    def get_documents_for_summarization(
            self
    ) -> List[Document]:
        """Return documents that are relevant to summarize."""
        current_time = datetime.now()
        two_weeks_ago = current_time - timedelta(weeks=2)
        filter_dict = {
            'must': {
                'metadata.last_accessed_at': {
                    'range': {'gte': two_weeks_ago.timestamp()}  # Filter for documents accessed at least 2 weeks ago
                }
            }
        }
        filter = self._qdrant_filter_from_dict(filter_dict)
        results = self.client.scroll(collection_name=self.collection_name, scroll_filter=filter, limit = 5000)
        docs = []
        for result in results:
            document = self.vectorstore._document_from_scored_point(
                result, self.vectorstore.content_payload_key, self.vectorstore.metadata_payload_key
            )

            # Increment the summarizations count
            if 'summarizations' in document.metadata:
                document.metadata['summarizations'] += 1
            else:
                document.metadata['summarizations'] = 1

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
        # Sort by score and extract just the documents
        sorted_docs = [doc for doc, _ in sorted(rescored_docs, key=lambda x: x[1], reverse=True)]
        # Return just the list of Documents
        return sorted_docs

    def clear_using_extra_index(self, extra_index) -> None:
        """Clear memory contents."""
        filter_dict = {
            'must': {
                'metadata.extra_index': {
                    'match': {'value': extra_index}
                }
            }
        }
        filter = self._qdrant_filter_from_dict(filter_dict)
        self.client.delete(collection_name=self.collection_name, points_selector=filter, wait = False)

    def does_key_exist(self, key, value):
        """Does the hash of the web content exist in our cache?."""
        filter_dict = {
            'must': {
                f'{key}': {
                    'match': {'value': value}
                },
            }
        }
        filter = self._qdrant_filter_from_dict(filter_dict)
        results = self.client.scroll(collection_name=self.collection_name, scroll_filter=filter, limit = 1)
        return results is not None and len(results[0]) > 0

    def prune_from(self, fromTime: float):
        """Prune points that are older than fromTime timestamp."""
        filter_dict = {
            'must': {
                'metadata.last_accessed_at': {
                    'range': {'gte': fromTime}
                }
            }
        }
        filter = self._qdrant_filter_from_dict(filter_dict)
        self.client.delete(collection_name=self.collection_name, points_selector=filter, wait = False)
