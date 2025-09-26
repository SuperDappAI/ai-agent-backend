from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from pydantic import Field, field_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.utils import get_from_dict_or_env
import logging

try:
    # Precise exception type for missing/unknown model errors
    from cohere.errors import NotFoundError as CohereNotFoundError
except Exception:  # pragma: no cover - fallback if import path changes
    CohereNotFoundError = Exception


@deprecated(
    since="0.0.30", removal="0.2.0", alternative_import="langchain_cohere.CohereRerank"
)
class CohereRerank(BaseDocumentCompressor):
    """Document compressor that uses `Cohere Rerank API`."""

    client: Any = None
    """Cohere client to use for compressing documents."""
    top_n: Optional[int] = 3
    """Number of documents to return."""

    # Default model updated to a current Cohere Rerank model. The older
    # "rerank-english-v2.0" returns 404 from the API as it has been
    # deprecated/removed. If needed, pass a model explicitly when
    # constructing this class or set via env from the caller.
    model: str = "rerank-v3.5"

    """Model to use for reranking."""
    cohere_api_key: Optional[str] = None
    """Cohere API key. Must be specified directly or via environment variable 
        COHERE_API_KEY."""
    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Initialize the client after model creation."""
        if not self.client:
            try:
                import cohere
            except ImportError:
                raise ImportError(
                    "Could not import cohere python package. "
                    "Please install it with `pip install cohere`."
                )
            cohere_api_key = get_from_dict_or_env(
                {"cohere_api_key": self.cohere_api_key}, "cohere_api_key", "COHERE_API_KEY"
            )
            self.client = cohere.AsyncClient(cohere_api_key, client_name=self.user_agent)

    async def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        model: Optional[str] = None,
        top_n: Optional[int] = -1,
        max_chunks_per_doc: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
            max_chunks_per_doc : The maximum number of chunks derived from a document.
        """  # noqa: E501
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = model or self.model
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        # First attempt with the provided/default model; if the model no longer
        # exists (404) fall back to Cohere's current recommended models.

        async def _call(target_model: str):
            return await self.client.rerank(
                query=query,
                documents=docs,
                model=target_model,
                top_n=top_n,
                max_chunks_per_doc=max_chunks_per_doc,
            )

        try_models: List[str] = [model]
        # Provide robust fallbacks in case of deprecations
        for candidate in ("rerank-english-v3.0", "rerank-multilingual-v3.0"):
            if candidate not in try_models:
                try_models.append(candidate)

        last_err: Optional[Exception] = None
        results = None
        for m in try_models:
            try:
                results = await _call(m)
                if m != model:
                    logging.warning(
                        f"CohereRerank: model '{model}' unavailable; fell back to '{m}'."
                    )
                break
            except CohereNotFoundError as e:  # model not found
                last_err = e
                logging.warning(
                    f"CohereRerank: model '{m}' not found (404). Trying next fallback if available..."
                )
            except Exception as e:
                # For non-404 issues, re-raise immediately
                raise

        if results is None and last_err is not None:
            # Exhausted fallbacks; raise the last NotFoundError
            raise last_err
        if hasattr(results, "results"):
            results = getattr(results, "results")
        result_dicts = []
        for res in results:
            result_dicts.append(
                {"index": res.index, "relevance_score": res.relevance_score}
            )
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Cohere's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        raise NotImplementedError()

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Cohere's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        # logging.info(f"acompress_documents: docs {documents} query {query}")
        for res in await self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(
                doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        # logging.info(f"acompress_documents: compressed {compressed} query {query}")
        return compressed
