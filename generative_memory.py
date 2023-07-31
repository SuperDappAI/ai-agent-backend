import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from time_weighted_retriever import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now
import json

logger = logging.getLogger(__name__)


class GenerativeAgentMemory(BaseMemory):
    """Memory for the generative agent."""

    llm: BaseLanguageModel
    """The core language model."""
    memory_retriever: TimeWeightedVectorStoreRetriever
    """The retriever to fetch related memories."""
    verbose: bool = False
    """How much weight to assign the memory importance."""
    
    # input keys
    queries_key: str = "queries"
    add_memory_user_key: str = "add_user_memory"
    add_memory_aida_key: str = "add_aida_memory"
    payload_conversation_key: str = "payload_conversation_key"
    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    now_key: str = "now"
    reflecting: bool = False

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def _get_topics_of_reflection(self, memory_content: str, conversation: str) -> [List[Document], List[str]]:
        """Return the 3 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given only the information above, what are the 3 most salient "
            "high-level questions we can answer about the subjects in the statements?\n"
            "Provide each question on a new line."
        )
        # get last important memories to get reflections on them
        observationsDocuments = self.memory_retriever.get_relevant_documents_for_reflection(memory_content, conversation)
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

    def pause_to_reflect(self, memory_content: str, conversation: str, now: Optional[datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("AiDA is reflecting")
        new_insights = []
        observationDocuments, topics = self._get_topics_of_reflection(memory_content, conversation)
        if len(observationDocuments) > 0 and len(topics) > 0:
            insights = self._get_insights_on_topics(topics, observationDocuments, conversation=conversation, now=now)
            if len(insights) > 0:
                qa = {"my_reflections": topics, "my_insights": insights}
                self.add_memory(json.dumps(qa), conversation, now=now)
                new_insights.extend(insights)
                return new_insights
        return []

    def _score_memory_importance(self, memory_content: str) -> int:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        score = self.chain(prompt).run(memory_content=memory_content).strip()
        if self.verbose:
            logger.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return int(match.group(1))
        else:
            return 0

    def _score_memories_importance(self, memory_list: str) -> List[float]:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Always answer with only a list of numbers."
            + " If just given one memory still respond in a list."
            + " Memories are separated by semi colans (;)"
            + "\Memories: {memory_list}"
            + "\nRating: "
        )
        scores = self.chain(prompt).run(memory_list=memory_list).strip()

        if self.verbose:
            logger.info(f"Importance scores: {scores}")

        # Split into list of strings and convert to floats
        scores_list = [float(x) for x in scores.split(";")]

        return scores_list

    def add_memories(
        self, qa: List[str], conversation: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observations or memories to the agent's memory."""
        memory_list = self.format_qa_simple(qa)
        importance_scores = self._score_memories_importance(memory_list)
        documents = []
        max_importance = 0

        for i in range(len(qa)):
            metadata = {
                "conversation": conversation,
                "created_at": now,
                "importance_score": importance_scores[i],
                "last_accessed_at": now
            }
            doc = Document(
                    page_content=qa[i],
                    metadata=metadata
                )
            documents.append(doc)
            if importance_scores[i] > max_importance:
                max_importance = importance_scores[i]
                max_importance_doc = doc
        
        result = self.memory_retriever.vectorstore.add_documents(documents)
        if ( max_importance >= 9):
            # reflect on the most important memory with like memories that were also important
            self.pause_to_reflect(max_importance_doc.page_content, conversation, now=now)
        return result

    def add_memory(
        self, memory_content: str, conversation: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        importance_score = self._score_memory_importance(memory_content)
        metadata = {
            "conversation": conversation,
            "created_at": now,
            "importance_score": importance_score, 
            "last_accessed_at": now,
        }
        document = Document(
            page_content=memory_content, 
            metadata=metadata,
        )
        result = self.memory_retriever.vectorstore.add_documents([document])
        if (importance_score >= 9):
            # reflect on the most important memory with like memories that were also important
            self.pause_to_reflect(memory_content, conversation, now=now)
        return result

    def fetch_memories(
        self, topic: str, **kwargs: Any
    ) -> List[Document]:
        """Fetch related memories."""
        current_time = kwargs.get("current_time", None)
        conversation = kwargs.get(self.payload_conversation_key)
        if current_time is not None:
            with mock_now(current_time):
                return self.memory_retriever.get_relevant_documents(topic)
        else:
            oldargs = self.memory_retriever.search_kwargs.copy()
            self.memory_retriever.search_kwargs.update({"filter": {"conversation": conversation}})
            docs = self.memory_retriever.get_relevant_documents(topic)
            self.memory_retriever.search_kwargs = oldargs
            return docs

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}[{created_time}] {memory.page_content.strip()}"

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    def format_qa_simple(self, qa: List[object]) -> str:
        return "; ".join(mem for mem in qa)

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        queries = inputs.get(self.queries_key)
        now = inputs.get(self.now_key)
        conversation = inputs.get(self.payload_conversation_key)
        kwargs = {self.payload_conversation_key: conversation, "current_time": now}
        if queries is not None:
            relevant_memories = [
                mem for query in queries for mem in self.fetch_memories(query, **kwargs)
            ]
            return {
                self.relevant_memories_key: self.format_memories_detail(
                    relevant_memories
                ),
                self.relevant_memories_simple_key: self.format_memories_simple(
                    relevant_memories
                ),
            }
        return {}

    def save_context(self, outputs: Dict[str, Any]) -> None:
        """Save the context of this model run to memory."""
        user = outputs.get(self.add_memory_user_key)
        aida = outputs.get(self.add_memory_aida_key)
        now = outputs.get(self.now_key)
        conversation = outputs.get(self.payload_conversation_key)
        if user:
            qa = {"user": user, "me": aida}
            self.add_memory(json.dumps(qa), conversation, now=now)

    def clear(self) -> None:
        """Clear memory contents."""
        # TODO