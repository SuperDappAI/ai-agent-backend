import logging
import json
import random
import traceback

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from qdrant_retriever import MemoryType
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now
from qdrant_client.http import models as rest
from memory_summarizer import MemorySummarizer
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)
    
class GenerativeAgentMemory(BaseMemory):
    rate_limiter: RateLimiter
    """Memory for the generative agent."""
    llm: BaseLanguageModel
    """The core language model."""
    memory_retriever: ContextualCompressionRetriever
    """The retriever to fetch related memories."""
    memory_summarizer: MemorySummarizer
    """Memory summarizer to be used when adding core memories."""
    verbose: bool = False

    @staticmethod
    def _extract_insights(text: str) -> List[str]:
        """Extract insights from the provided text."""
        # Split the text into lines
        lines = text.splitlines()
        # Find the index of the "Insights:" line
        try:
            start_idx = lines.index("Insights: ") + 1
        except ValueError:
            try:
                start_idx = lines.index("Insights:") + 1
            except ValueError:
                return []
        # Extract insights until an empty line or end of text
        insights = []
        for line in lines[start_idx:]:
            if not line.strip():
                break
            insights.append(line.strip())
        insights_str = '\n'.join(insights)
        return insights_str

    @staticmethod
    def _extract_importance(text: str) -> str:
        """Extract importance level from the provided text."""
        
        # Split the text into lines
        lines = text.splitlines()
        
        # Find the line with "Importance:"
        for line in lines:
            if "Importance:" in line:
                # Extract the importance level
                level = line.split("Importance:")[1].strip()
                return level.lower()

        return "low"

    def format_memories_as_messages(self, relevant_memories: List[Document]) -> List[BaseMessage]:
        formatted_memories = []
        for mem in relevant_memories:
            memory = json.loads(mem.page_content)  # Convert the JSON string to a dictionary
            formatted_memories.append(HumanMessage(content=f'Memory: {memory["user"]}'))
            formatted_memories.append(AIMessage(content=f'Memory: {memory["AiDA"]}'))
        return formatted_memories

    async def _get_importance_and_insight(self, user: str, llm_response: str, conversation_id: str, role: str):
        """Reflect on recent query and generate 'insights'."""
        if self.verbose:
            logger.info("AiDA is checking importance")
        kwargs = {}
        # lookup some relevant context for query classification
        memoryDocuments = await self.memory_retriever.base_retriever.get_relevant_documents_for_reflection(json.dumps({'user': user, 'AiDA': llm_response}), conversation_id, **kwargs)
        try:
            memoryMessages = self.format_memories_as_messages(memoryDocuments)
            prompt = f"""
                Based on your role ({role}) analyze the given dialog and relevant memories between the user and AiDA (an AI assistant), classify dialog in terms of importance from low, medium, or high and then provide questions and infer novel insights.
                Determining importance: 'low' are ones that can usually be forgotten. The default is 'medium' memories which are useful to remember long-term. 'high' are useful memories to remember and have sufficient context and information to be able to derive multiple new insights and topics.
                If importance is 'low' or 'medium' just output the importance and not any questions or insights.
                If it is 'high', answer the question: What are the 3 most salient high-level questions we can answer about the subjects in the statements (Provide each question on a new line)?
                Also answer the question: What are 3 high-level novel insights that are relevant for answering the 3 high-level questions? (Provide each insight on a new line)
                Do not include any insights that are not relevant to the questions.
                Be concise.
                Do not repeat any insights that have already been made.
                
                Format:
                
                Importance: <level, low/medium/high>
                Questions: <questions, each on a new line>
                Insights: <insights, each on a new line>
                """
            messages = [SystemMessage(content=prompt)]
            messages.extend(memoryMessages)  # Extend the list with memoryMessages
            messages.append(HumanMessage(content=user))
            messages.append(AIMessage(content=llm_response))
            response = await self.llm.agenerate([messages])
            if not response.generations or not response.generations[0]:
                raise Exception("LLM did not provide a valid summary response.")
            result = response.generations[0][0].text
            importance = self._extract_importance(result)
            insights = self._extract_insights(result)
            return importance, insights
        except Exception as e:
            if self.verbose:
                logging.warn(f"GenerativeAgentMemory: _get_importance_and_insight exception, e: {e}\n{traceback.format_exc()}")
            return None, None

    async def pause_to_reflect(self, outputs: Dict[str, Any], preferences_resolver) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        new_insights = []
        conversation_id = outputs.get("conversation_id")
        query = outputs.get("query")
        aida = outputs.get("llm_response")
        now=datetime.now()
        try:
            role = await preferences_resolver.get_role(conversation_id)
            if role is None:
                role = "ConversationGPT"
            importance, insights = await self._get_importance_and_insight(query, aida, conversation_id, role)
            if importance == "high" and len(insights) > 0:
                if self.verbose:
                    logger.info("AiDA is reflecting")
                # ensure we are dealing with non-core memories because reflections are sub-conscious thoughts
                await self.add_memory(memory_content=json.dumps({'user': 'AiDA to reflect and generate insight', 'AiDA': insights}), conversation_id=conversation_id, importance="medium", memory_type=MemoryType.SUBCONSCIOUS_MEMORY, now=now)
                new_insights.extend(insights)
        except Exception as e:
            importance = 'low'
            if self.verbose:
                logging.warn(f"GenerativeAgentMemory: pause_to_reflect exception, e: {e}\n{traceback.format_exc()}")           
        outputs["importance"] = importance
        await self.save_context(outputs)
        return new_insights

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
        return await self.rate_limiter.execute(self.memory_retriever.base_retriever.vectorstore.aadd_documents, documents, ids=ids)

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
        return await self.rate_limiter.execute(self.memory_retriever.base_retriever.vectorstore.aadd_documents, [document], ids=[metadata["id"]])

    async def fetch_memories(
        self, topic: str, **kwargs: Any
    ) -> List[Document]:
        """Fetch related memories."""
        current_time = kwargs.get("current_time", None)
        conversation_id = kwargs.pop("conversation_id")
        if current_time is not None:
            with mock_now(current_time):
                return await self.memory_retriever.ainvoke(topic)
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
            return await self.memory_retriever.ainvoke(topic, **kwargs)

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

    async def load_memory_variables(self, **kwargs) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        queries = kwargs.pop("queries")
        if queries is not None:
            relevant_memories = [
                mem for query in queries for mem in await self.fetch_memories(query, **kwargs)
            ]
            if len(relevant_memories) > 0:
                # update last_accessed_at/summarizations
                ids = [doc.metadata["id"] for doc in relevant_memories]
                for doc in relevant_memories:
                    doc.metadata.pop('relevance_score', None)
                await self.rate_limiter.execute(self.memory_retriever.base_retriever.vectorstore.aadd_documents, relevant_memories, ids=ids)
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
            qa = {'user': query, 'AiDA': aida}
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
                await self.rate_limiter.execute(self.memory_retriever.base_retriever.vectorstore.aadd_documents, documents, ids=ids) 
        except Exception as e:
            logging.warn(f"GenerativeAgentMemory: decay_user exception {e}\n{traceback.format_exc()}")
            
    def clear(self) -> None:
        return