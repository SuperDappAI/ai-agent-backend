import logging
import re
import json
import random
import asyncio

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from qdrant_retriever import MemoryType
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now
from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)
    
class GenerativeAgentMemory(BaseMemory):
    """Memory for the generative agent."""

    llm: BaseLanguageModel
    """The core language model."""
    memory_retriever: ContextualCompressionRetriever
    """The retriever to fetch related memories."""
    verbose: bool = False

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def _get_topics_of_reflection(self, memory_content: str, user_id: str, conversation: str) -> [List[Document], List[str]]:
        """Return the 3 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given only the information above, what are the 3 most salient "
            "high-level questions we can answer about the subjects in the statements?\n"
            "Provide each question on a new line."
        )
        # get last important memories to get reflections on them
        kwargs = {"score_threshold": 0.6, "k": 11}
        observationsDocuments = self.memory_retriever.base_retriever.get_relevant_documents_for_reflection(memory_content, user_id, conversation, **kwargs)
        if len(observationsDocuments) > 0:
            observation_str = "\n".join(
                [self._format_memory_detail(o) for o in observationsDocuments]
            )
            result = self.chain(prompt).run(observations=observation_str)
            return observationsDocuments, self._parse_list(result)
        else:
            return [], []

    def _get_insights_on_topics(
        self, topics: List[str], observationDocuments: List[Document], **kwargs: Any,
    ) -> List[str]:
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            "Related statements to questions:\n"
            "---\n"
            "{related_statements}\n"
            "---\n"
            "What 5 high-level novel insights can you infer from the above statements "
            "that are relevant for answering the following questions?\n"
            "Do not include any insights that are not relevant to the questions.\n"
            "Do not repeat any insights that have already been made.\n\n"
            "Questions: {topics}\n\n"
            "(example format: insight (because of 1, 3))\n"
        )
        related_statements = "\n".join(
            [
                self._format_memory_detail(observation, prefix=f"{i+1}. ")
                for i, observation in enumerate(observationDocuments)
            ]
        )
        result = self.chain(prompt).run(
            topics=topics, related_statements=related_statements
        )
        return self._parse_list(result)

    async def pause_to_reflect(self, memory_content: str, user_id: str, conversation: str, now: Optional[datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("AiDA is reflecting")
        new_insights = []
        observationDocuments, topics = self._get_topics_of_reflection(memory_content, user_id, conversation)
        if len(observationDocuments) > 0 and len(topics) > 0:
            insights = self._get_insights_on_topics(topics, observationDocuments, conversation=conversation, now=now)
            if len(insights) > 0:
                qa = {"my_reflections": topics, "my_insights": insights}
                # ensure we are dealing with non-core memories because reflections are sub-conscious thoughts
                await self.add_memory(memory_content=json.dumps(qa), user_id=user_id, conversation_id=conversation, importance="medium", memory_type=MemoryType.SUBCONSCIOUS_MEMORY, now=now)
                new_insights.extend(insights)
                return new_insights
        return []

    async def add_memories(
        self, qa: List[str], user_id: str, conversation_id: str, importance: List[str], memory_types: List[MemoryType], now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observations or memories to the agent's memory."""
        documents = []
        ids = []
        nowStamp = now.timestamp()
        for i in range(len(qa)):
            metadata = {
                "id":  random.randint(0, 2**32 - 1),
                "extra_index": conversation_id,
                "created_at": nowStamp,
                "importance": importance[i],
                "last_accessed_at": nowStamp,
                "summarizations": 0,
                "group_id": user_id,
                "memory_type": memory_types[i].value,
            }
            doc = Document(
                    page_content=qa[i],
                    metadata=metadata
                )
            documents.append(doc)
            ids.append(metadata["id"])
        
        return await self.memory_retriever.base_retriever.vectorstore.aadd_documents(documents, ids=ids, wait = False)

    async def add_memory(
        self, memory_content: str, user_id: str, conversation_id: str, importance: str, memory_type: MemoryType, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        nowStamp = now.timestamp()
        metadata = {
            "id": random.randint(0, 2**32 - 1),
            "extra_index": conversation_id,
            "created_at": nowStamp,
            "importance": importance, 
            "last_accessed_at": nowStamp,
            "summarizations": 0,
            "group_id": user_id,
            "memory_type": memory_type.value,
        }
        document = Document(
            page_content=memory_content, 
            metadata=metadata,
        )
        return await self.memory_retriever.base_retriever.vectorstore.aadd_documents([document], ids=[metadata["id"]], wait = False)

    def fetch_memories(
        self, topic: str, **kwargs: Any
    ) -> List[Document]:
        """Fetch related memories."""
        current_time = kwargs.get("current_time", None)
        conversation_id = kwargs.pop("conversation_id")
        if current_time is not None:
            with mock_now(current_time):
                return self.memory_retriever.get_relevant_documents(topic)
        else:
            if conversation_id is not "":
                kwargs.update({"filter": rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.extra_index", 
                            match=rest.MatchValue(value=conversation_id), 
                        )
                    ]
                )})
            docs = self.memory_retriever.get_relevant_documents(topic, **kwargs)
            return docs

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        memory_type = MemoryType(memory.metadata["memory_type"]).name.replace("_", " ").lower()
        created_time = datetime.fromtimestamp(memory.metadata["created_at"]).strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}[{created_time}] ({memory_type}) {memory.page_content.strip()}"

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

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        now = datetime.now().timestamp()
        formatted_memories = []
        for mem in relevant_memories:
            memory_type = MemoryType(mem.metadata["memory_type"]).name.replace("_", " ").lower()
            summarizations_count = mem.metadata.get("summarizations", 0)
            importance = mem.metadata.get("importance", "medium")  # assuming "medium" as default importance
            created_at = mem.metadata.get("created_at", now)
            created_ago = self._time_ago(created_at)
            
            # Extracting the extra_index (conversation_id)
            conversation_id = mem.metadata.get("extra_index", "N/A")
            formatted_memories.append(f"({memory_type}, importance: {importance}, summarizations: {summarizations_count}, from: {created_ago}, conversation_id: {conversation_id}) {mem.page_content}")
        return "; ".join(formatted_memories)

    def format_qa_simple(self, qa: List[object]) -> str:
        return "; ".join(mem for mem in qa)

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return []

    def load_memory_variables(self, **kwargs) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        queries = kwargs.pop("queries")
        if queries is not None:
            relevant_memories = [
                mem for query in queries for mem in self.fetch_memories(query, **kwargs)
            ]
            ids = [doc.metadata["id"] for doc in relevant_memories]
            for doc in relevant_memories:
                doc.metadata.pop('relevance_score', None)
            asyncio.create_task(self.memory_retriever.base_retriever.vectorstore.aadd_documents(relevant_memories, ids=ids, wait = False))
            return {
                "relevant_memories": self.format_memories_simple(relevant_memories),
            }
        return {}

    async def save_context(self, outputs: Dict[str, Any]) -> List[str]:
        """Save the context of this model run to memory."""
        query = outputs.get("query")
        aida = outputs.get("llm_response")
        now = datetime.now()
        importance = outputs.get("importance")
        conversation_id = outputs.get("conversation_id")
        user_id = outputs.get("user_id")
        if query:
            qa = {"user": query, "me": aida}
            return await self.add_memory(json.dumps(qa), user_id=user_id, conversation_id=conversation_id, memory_type=MemoryType.CONSCIOUS_MEMORY, importance=importance, now=now)
        return []

    def clear(self, conversation_id) -> None:
        """Clear memory contents."""
        self.memory_retriever.base_retriever.clear_using_extra_index(conversation_id)