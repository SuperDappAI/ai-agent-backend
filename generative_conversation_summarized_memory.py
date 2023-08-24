import logging
import random

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from qdrant_retriever import MemoryType
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel

logger = logging.getLogger(__name__)
    
class GenerativeAgentConversationSummarizedMemory(BaseMemory):
    """Conversations summarized for the generative agent."""

    llm: BaseLanguageModel
    """The core language model."""
    memory_retriever: ContextualCompressionRetriever
    """The retriever to fetch related memories."""
    verbose: bool = False

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def _init_summary_of_convo(self, doc0: str) -> str:
        prompt = PromptTemplate.from_template(
            "{doc0}\n\n"
            "Create a topic-based summarization of only text above. If it is above 1000 words summarize to stay below. Be concise, do not add new details that is not in the provided text. Output is a new JSON object. example {{\"topic\",\"topic summary\"}}."
        )
        return self.chain(prompt).run(doc0=doc0)

    def _summarize_with_convo(self, new_text: str, existing_summary: str) -> str:
        prompt = PromptTemplate.from_template(
            "{existing_summary}\n\n"
            "{new_text}\n\n"
            "Combine second text into the summary (first text) and return new summary using only the information above. Remove any redundancies. Create new topics if any of the information does not belong to existing topics. If result is above 1000 words summarize to stay below. Be concise, do not add new details that is not in the provided text or in the existing summary. Output is the modified JSON object."
        )
        return self.chain(prompt).run(existing_summary=existing_summary, new_text=new_text)

    async def add_memories(
        self, qa: List[str], conversation_id: str, importance: List[str], memory_types: List[MemoryType], now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observations or memories to the agent's memory."""
        documents = []
        ids = []
        nowStamp = now.timestamp()
        for i in range(len(qa)):
            if memory_types[i] != MemoryType.CONSCIOUS_MEMORY or importance[i] == "low":
                continue
            metadata = {
                "id":  random.randint(0, 2**32 - 1),
                "extra_index": conversation_id,
                "created_at": nowStamp,
            }
            doc = Document(
                page_content=qa[i],
                metadata=metadata
            )
            documents.append(doc)
            ids.append(metadata["id"])
        if len(documents) > 0:
            return await self.memory_retriever.base_retriever.vectorstore.aadd_documents(documents, ids=ids, wait = False)
        else:
            return None

    # if we are dealing with important conscious thoughts then we can create personality and goals/subgoals (intentions)
    # add into global memory as well with learnings that can apply into AiDAs personality (traits, mood, feelings) and goals/subgoals
    # as well as users personality and goals (what AiDA thinks of the user and the goals/subgoals)  
    async def add_memory(
        self, memory_content: str, conversation_id: str, importance: str, memory_type: MemoryType, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        if memory_type != MemoryType.CONSCIOUS_MEMORY or importance == "low":
            return None
        nowStamp = now.timestamp()
        metadata = {
            "id": random.randint(0, 2**32 - 1),
            "extra_index": conversation_id,
            "created_at": nowStamp,
        }
        document = Document(
            page_content=memory_content, 
            metadata=metadata,
        )
        # pull existing conversation and merge the two into a new memory
        doc = self.get_conversation(conversation_id)
        if doc is not None:
            # summarize the two together
            document.metadata = doc.metadata
            document.page_content = self._summarize_with_convo(document.page_content, doc.page_content)
        else:
            document.page_content = self._init_summary_of_convo(document.page_content)
        return await self.memory_retriever.base_retriever.vectorstore.aadd_documents([document], ids=[document.metadata["id"]], wait = False)

    async def save_context(self, outputs: Dict[str, Any]) -> List[str]:
        """Save the context of this model run to memory."""
        query = outputs.get("query")
        aida = outputs.get("llm_response")
        now = datetime.now()
        importance = outputs.get("importance")
        conversation_id = outputs.get("conversation_id")
        if query:
            return await self.add_memory(aida,  conversation_id=conversation_id, memory_type=MemoryType.CONSCIOUS_MEMORY, importance=importance, now=now)
        return []
    
    def get_conversation(
        self, conversation_id: str
    ) -> Document:
        """Fetch summarized conversation."""
        return self.memory_retriever.base_retriever.get_key_value_document("metadata.extra_index", conversation_id)

    def _time_ago(self, timestamp: float) -> str:
        """Return a rough string representation of the time passed since a timestamp."""
        delta = datetime.now() - datetime.fromtimestamp(timestamp)
        if delta < timedelta(minutes=1):
            return "just now"
        elif delta < timedelta(hours=1):
            return f"{int(delta.total_seconds() / 60)} minutes ago"
        elif delta < timedelta(days=1):
            return f"{int(delta.total_seconds() / 3600)} hours ago"
        else:
            return f"{int(delta.total_seconds() / 86400)} days ago"

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return []
    
    def clear(self) -> None:
        return

    def load_memory_variables(self, **kwargs) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        conversation_id = kwargs.pop("conversation_id")
        return {
            "relevant_summary": self.format_summary_simple(self.get_conversation(conversation_id)),
        }
    
    def format_summary_simple(self, conversation_summary: Document) -> str:
        now = datetime.now().timestamp()
        created_at = conversation_summary.metadata.get("created_at", now)
        created_ago = self._time_ago(created_at)
        
        # Extracting the extra_index (conversation_id)
        conversation_id = conversation_summary.metadata.get("extra_index", "N/A")
        return f"(created: {created_ago}, conversation_id: {conversation_id}) {conversation_summary.page_content}"