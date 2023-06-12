from typing import Any, Dict, List, Optional, Union

from pydantic import Field
from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from abc import ABC, abstractmethod
from pydantic import BaseModel, Extra, Field, root_validator
import threading
import asyncio

class CustomMemory(BaseModel, ABC):
    """Base interface for memory in chains."""
    class Config:
        """Configuration for this pydantic object."""
        
        extra = Extra.allow
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain.

        If None, return all memories
        """

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""


class VectorStoreRetrieverMemoryBuffer(CustomMemory):
    """Class for a VectorStore-backed memory object."""
    retriever: VectorStoreRetriever = Field(exclude=True)
    memory_key: str = "history"
    input_key: Optional[str] = None

    return_docs: bool = False
    batch_size: int = 32 
    _message_buffer: List[Document] = []
    _buffer_lock: threading.Lock = threading.Lock()

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def load_memory_variables(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, Union[List[Document], str]]:
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.get_relevant_documents(query)
        result: Union[List[Document], str]
        if not self.return_docs:
            result = "\n".join([doc.page_content for doc in docs])
        else:
            result = docs
        return {self.memory_key: result}

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        filtered_inputs = {k: v for k, v in inputs.items() if k != self.memory_key}
        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]

    async def background_upsert(self, documents: List[Document]) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.retriever.add_documents, documents)

    async def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        documents = self._form_documents(inputs, outputs)

        with self._buffer_lock:
            self._message_buffer.extend(documents)
            buffer_len = len(self._message_buffer)

        if buffer_len >= self.batch_size:
            with self._buffer_lock:
                batch, self._message_buffer = self._message_buffer[: self.batch_size], self._message_buffer[self.batch_size:]
            await self.background_upsert(batch)

    def clear(self) -> None:
        """Nothing to clear."""


# from typing import Any, Dict, List, Optional, Union

# from pydantic import Field

# from langchain.memory.chat_memory import BaseMemory
# from langchain.memory.utils import get_prompt_input_key
# from langchain.schema import Document
# from langchain.vectorstores.base import VectorStoreRetriever
# from abc import ABC, abstractmethod
# from pydantic import BaseModel, Extra, Field, root_validator

# class CustomMemory(BaseModel, ABC):
#     """Base interface for memory in chains."""
#     class Config:
#         """Configuration for this pydantic object."""
        
#         extra = Extra.allow
#         arbitrary_types_allowed = True

#     @property
#     @abstractmethod
#     def memory_variables(self) -> List[str]:
#         """Input keys this memory class will load dynamically."""

#     @abstractmethod
#     def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
#         """Return key-value pairs given the text input to the chain.

#         If None, return all memories
#         """

#     @abstractmethod
#     def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
#         """Save the context of this model run to memory."""

#     @abstractmethod
#     def clear(self) -> None:
#         """Clear memory contents."""


# class VectorStoreRetrieverMemoryBuffer(CustomMemory):
#     """Class for a VectorStore-backed memory object."""

#     retriever: VectorStoreRetriever = Field(exclude=True)
#     memory_key: str = "history"
#     input_key: Optional[str] = None

#     return_docs: bool = False
#     batch_size: int = 32 
#     _message_buffer: List[Document] = []

#     @property
#     def memory_variables(self) -> List[str]:
#         return [self.memory_key]

#     def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
#         if self.input_key is None:
#             return get_prompt_input_key(inputs, self.memory_variables)
#         return self.input_key

#     def load_memory_variables(
#         self, inputs: Dict[str, Any]
#     ) -> Dict[str, Union[List[Document], str]]:
#         input_key = self._get_prompt_input_key(inputs)
#         query = inputs[input_key]
#         docs = self.retriever.get_relevant_documents(query)
#         result: Union[List[Document], str]
#         if not self.return_docs:
#             result = "\n".join([doc.page_content for doc in docs])
#         else:
#             result = docs
#         return {self.memory_key: result}

#     def _form_documents(
#         self, inputs: Dict[str, Any], outputs: Dict[str, str]
#     ) -> List[Document]:
#         filtered_inputs = {k: v for k, v in inputs.items() if k != self.memory_key}
#         texts = [
#             f"{k}: {v}"
#             for k, v in list(filtered_inputs.items()) + list(outputs.items())
#         ]
#         page_content = "\n".join(texts)
#         return [Document(page_content=page_content)]

#     async def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
#         documents = self._form_documents(inputs, outputs)
#         self._message_buffer.extend(documents)

#         if len(self._message_buffer) >= self.batch_size:
#             self.retriever.add_documents(self._message_buffer[: self.batch_size])
#             self._message_buffer = self._message_buffer[self.batch_size :]

#     def clear(self) -> None:
#         """Nothing to clear."""
