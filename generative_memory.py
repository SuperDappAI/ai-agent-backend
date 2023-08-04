import logging
import re
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now
import json

logger = logging.getLogger(__name__)


class GenerativeAgentMemory(BaseMemory):
    """Memory for the generative agent."""

    llm: BaseLanguageModel
    """The core language model."""
    memory_retriever: QDrantVectorStoreRetriever
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

    def _get_topics_of_reflection(self, id_to_skip: str, memory_content: str, user_id: str, conversation: str) -> [List[Document], List[str]]:
        """Return the 3 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given only the information above, what are the 3 most salient "
            "high-level questions we can answer about the subjects in the statements?\n"
            "Provide each question on a new line."
        )
        # get last important memories to get reflections on them
        kwargs = {"score_threshold": 0.8, "k": 10}
        observationsDocuments = self.memory_retriever.get_relevant_documents_for_reflection(id_to_skip, memory_content, user_id, conversation, **kwargs)
        print(f"_get_topics_of_reflection observationsDocuments {observationsDocuments}")
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

    async def pause_to_reflect(self, id_to_skip: str, memory_content: str, user_id: str, conversation: str, now: Optional[datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("AiDA is reflecting")
        new_insights = []
        observationDocuments, topics = self._get_topics_of_reflection(id_to_skip, memory_content, user_id, conversation)
        if len(observationDocuments) > 0 and len(topics) > 0:
            insights = self._get_insights_on_topics(topics, observationDocuments, conversation=conversation, now=now)
            print(f"_get_insights_on_topics {insights}")
            if len(insights) > 0:
                qa = {"my_reflections": topics, "my_insights": insights}
                # ensure we are dealing with non-core memories because reflections are sub-conscious thoughts
                asyncio.create_task(self.add_memory(json.dumps(qa), conversation, importance_score=8 , now=now))
                new_insights.extend(insights)
                return new_insights
        return []

    async def add_memories(
        self, qa: List[str], user_id: str, conversation: str, importance_scores: List[int], now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observations or memories to the agent's memory."""
        documents = []
        nowStamp = now.timestamp()
        for i in range(len(qa)):
            metadata = {
                "extra_index": conversation,
                "created_at": nowStamp,
                "importance_score": importance_scores[i],
                "last_accessed_at": nowStamp,
                "summarizations": 0,
                "group_id": user_id,
            }
            doc = Document(
                    page_content=qa[i],
                    metadata=metadata
                )
            documents.append(doc)
        
        return await self.memory_retriever.vectorstore.aadd_documents(documents, wait = False)

    async def add_memory(
        self, memory_content: str, user_id: str, conversation: str, importance_score: int, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        nowStamp = now.timestamp()
        metadata = {
            "extra_index": conversation,
            "created_at": nowStamp,
            "importance_score": importance_score, 
            "last_accessed_at": nowStamp,
            "summarizations": 0,
            "group_id": user_id,
        }
        document = Document(
            page_content=memory_content, 
            metadata=metadata,
        )
        return await self.memory_retriever.vectorstore.aadd_documents([document], wait = False)

    def fetch_memories(
        self, topic: str, **kwargs: Any
    ) -> List[Document]:
        """Fetch related memories."""
        current_time = kwargs.get("current_time", None)
        conversation_id = kwargs.pop("conversation_id")
        if current_time is not None:
            print(f"fetch_memories kwargs current_time")
            with mock_now(current_time):
                return self.memory_retriever.get_relevant_documents(topic)
        else:
            filter_dict = {
                'must': {
                    'metadata.extra_index': {
                        'match': {'value': conversation_id}
                    }
                }
            }
            filter = self.memory_retriever._qdrant_filter_from_dict(filter_dict)
            kwargs.update({"filter": filter})
            docs = self.memory_retriever.get_relevant_documents(topic, **kwargs)
            return docs

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = datetime.fromtimestamp(memory.metadata["created_at"]).strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}[{created_time}] {memory.page_content.strip()}"

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

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
            return {
                "relevant_memories": self.format_memories_detail(relevant_memories),
                "relevant_memories_simple": self.format_memories_simple(relevant_memories),
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
            qa = {user_id: query, "me": aida}
            return await self.add_memory(json.dumps(qa), user_id, conversation_id, importance, now=now)
        return []

    def clear(self, conversation_id) -> None:
        """Clear memory contents."""
        self.memory_retriever.clear_using_extra_index(conversation_id)