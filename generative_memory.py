import logging
import re
import json
import random
import asyncio
import traceback

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
from memory_summarizer import MemorySummarizer
from personality_resolver import PersonalityResolver

logger = logging.getLogger(__name__)
    
class GenerativeAgentMemory(BaseMemory):
    """Memory for the generative agent."""
    personality_resolver: PersonalityResolver
    llm: BaseLanguageModel
    """The core language model."""
    memory_retriever: ContextualCompressionRetriever
    """The retriever to fetch related memories."""
    memory_summarizer: MemorySummarizer
    """Memory summarizer to be used when adding core memories."""
    verbose: bool = False

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
        kwargs = {}
        observationsDocuments = self.memory_retriever.base_retriever.get_relevant_documents_for_reflection(memory_content, conversation, **kwargs)
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

    def _get_json_patch_commands(
        self, conversation: str, personality,
    ) -> List[str]:
        """Generate 'personality updates', based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            "Personality:\n"
            "---\n"
            "{personality}\n"
            "---\n"
            "You're a top-tier personality interpreter. Your task is to read the conversation and the given personality, then suggest adjustments to attributes.\n"
            "Personality is in context of the user and is useful across conversations; are provided in context of every exchange with AI. Things like name/nickname, moods, goals, tasks, accomplishments are common examples for users personality attributes.\n"
            "List inferred personality update topics, keeping them broad yet meaningful.\n"
            "Skip updates for code-related or non-conversational exchanges.\n"
            "Only include conversation-relevant changes. If unsure, return an empty array.\n"
            "First, identify up to 3 broad changes. Then, provide an array of JSON patch commands ('add', 'remove', 'replace'). Format:  '/traits/-' to add, '/traits/[some_integer]' to remove/replace at specific index.\n"
            "The personality schema is not static, you may adjust it as needed. Add/remove fields/subfields at your descretion. 'Tasks' schema should remain intact as we depend on the structure defined.\n"
            "Be aware of token limits, calculate the token count for personality above and estimated changes with the outputted list and try to keep the total size up to 1000 tokens, remove redundant attributes if needed. Tasks and goals are highest priority.\n"
            "Importantly, triple check that you format the output correctly (use given examples for reference).\n"
            "Avoid duplicating previous adjustments.\n\n"
            "Conversation: {conversation}\n\n"
            "Examples: \n"
            "- Description: Nothing found\n"
            "Output: []\n"
            "- Description: Addressing multiple changes\n"
            'Output: [{"op": "add", "path": "/traits/-", "value": "adventurous"},{"op": "remove", "path": "/traits/1"},{"op": "replace", "path": "/traits/0", "value": "meticulous"}]\n'
            "- Description: Updating task and subtask\n"
            'Output: [{"op": "add", "path": "/tasks/-", "value": {"task": "New Task", "active": false},{"op": "add", "path": "/tasks/0/subtasks/-", "value": {"subtask": "New Subtask", "active": false},{"op": "replace", "path": "/tasks/0/active", "value": true},{"op": "replace", "path": "/tasks/0/subtasks/0/active", "value": true}]\n'
            "- Description: Multiple updates in various areas\n"
            'Output: [{"op": "add", "path": "/achievements/-", "value": "New Achievement"},{"op": "add", "path": "/expertise/-", "value": "New Skill"},{"op": "replace", "path": "/mood_feelings/0", "value": "content"}]\n'
            "- Description: Changing privacy settings\n"
            'Output: [{"op": "replace", "path": "/privacy/data_sharing/personal", "value": true}]\n'
            "- Description: Adding a new nickname\n"
            'Output: [{"op": "add", "path": "/name_nickname/-", "value": "JohnDoe"}]\n'
            "- Description: Removing a goal\n"
            'Output: [{"op": "remove", "path": "/goals/0"}]\n'
            "- Description: Changing mood and feelings\n"
            'Output: [{"op": "replace", "path": "/mood_feelings/0", "value": "sad"}]\n'
            "- Description: Adding new expertise\n"
            'Output: [{"op": "add", "path": "/expertise/-", "value": "Data Science"}]\n'
            "- Description: Removing an occupation\n"
            'Output: [{"op": "remove", "path": "/occupations/0"}]\n'
            "- Description: Adding facts and opinions\n"
            'Output: [{"op": "add", "path": "/facts_opinions/-", "value": "The earth is round"}]\n'
        )
        result = self.chain(prompt).run(
            personality=json.dumps(personality), conversation=conversation
        )
        return result

    async def pause_to_reflect(self, memory_content: str, conversation_id: str) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("AiDA is reflecting")
        new_insights = []
        now=datetime.now()
        observationDocuments, topics = self._get_topics_of_reflection(memory_content, conversation_id)
        if len(observationDocuments) > 0 and len(topics) > 0:
            insights = self._get_insights_on_topics(topics, observationDocuments, conversation=conversation_id, now=now)
            if len(insights) > 0:
                qa = {"my_reflections": topics, "my_insights": insights}
                # ensure we are dealing with non-core memories because reflections are sub-conscious thoughts
                await self.add_memory(memory_content=json.dumps(qa), conversation_id=conversation_id, importance="medium", memory_type=MemoryType.SUBCONSCIOUS_MEMORY, now=now)
                new_insights.extend(insights)
                return new_insights
        return []

    async def update_personality(self, memory_content: str, user_id: str):
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("AiDA is trying to update personality")
        doc = self.personality_resolver.get_personality(user_id)
        if doc is None:
            logger.warn(f"get_personality got empty doc")
            return
        patch_commands = self._get_json_patch_commands(memory_content, doc)
        print(f'patch_commands {patch_commands}')
        if len(patch_commands) > 0:
            response = self.personality_resolver.apply_patch(user_id, doc, patch_commands)
            if response != "success":
                logger.warn(f"personality_resolver patch application failed: {response}")
            else:
                logger.warn(f"update_personality success!")

    async def add_memories(
        self, qa: List[str], conversation_id: str, importance: List[str], memory_types: List[MemoryType], now: Optional[datetime] = None
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
        self, memory_content: str, conversation_id: str, importance: str, memory_type: MemoryType, now: Optional[datetime] = None
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
            if conversation_id != "":
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
            # update last_accessed_at/summarizations
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
        api_key = outputs.get("api_key")
        if query:
            qa = {"user": query, "AiDA": aida}
            await self.memory_summarizer.save(api_key, user_id, outputs)
            return await self.add_memory(json.dumps(qa), conversation_id=conversation_id, memory_type=MemoryType.CONSCIOUS_MEMORY, importance=importance, now=now)
        return []


    async def decay(self):
        """Decay all old memories by summarizing based on importance and summarization count."""
        try:
            # Delete memories flagged as too old
            self.memory_retriever.base_retriever.delete_max_summarized()
            # Get the documents to summarize
            documents = self.memory_retriever.base_retriever.get_documents_for_summarization()
            if len(documents) > 0:
                await self.memory_summarizer.flexible_document_summarizer.asummarize(documents)
                # upsert entire document set to qdrant against existing IDs (stored in metadata)
                ids = [doc.metadata["id"] for doc in documents]
                await self.memory_retriever.base_retriever.vectorstore.aadd_documents(documents, ids=ids) 
        except Exception as e:
            logging.warn(f"GenerativeAgentMemory: decay_user exception {e}\n{traceback.format_exc()}")
            
    def clear(self) -> None:
        return