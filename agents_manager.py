import time
import logging
import traceback
from dotenv import load_dotenv
import os
from datetime import datetime
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from asyncio import Lock
from rate_limiter import RateLimiter
import json
import websockets


class AgentListInput(BaseModel):
    user_id: str = None
    conversation_id: str = None
    results_page: int = 0
    agent_handle: str = None
    filter: str = None
    
    def __str__(self):
        if self.conversation_id:
            if self.filter:
                return self.conversation_id + self.filter + str(self.results_page)
            elif self.agent_handle:
                return self.conversation_id + self.agent_handle + str(self.results_page)
            else:
                return self.conversation_id + str(self.results_page)
        elif self.user_id:
            if self.filter:
                return self.user_id + self.filter + str(self.results_page)
            elif self.agent_handle:
                return self.user_id + self.agent_handle + str(self.results_page)
            else:
                return self.user_id + str(self.results_page)

    def __eq__(self, other):
        if self.conversation_id:
            if self.filter:
                return self.conversation_id  == other.conversation_id and self.filter == other.filter and self.results_page == other.results_page
            elif self.agent_handle:
                return self.conversation_id  == other.conversation_id and self.agent_handle == other.agent_handle and self.results_page == other.results_page
            else:
                return self.conversation_id  == other.conversation_id and self.results_page == other.results_page
        elif self.user_id:
            if self.filter:
                return self.user_id  == other.user_id and self.filter == other.filter and self.results_page == other.results_page
            elif self.agent_handle:
                return self.user_id  == other.user_id and self.agent_handle == other.agent_handle and self.results_page == other.results_page
            else:
                return self.user_id  == other.user_id and self.results_page == other.results_page

    def __hash__(self):
        return hash(str(self))

class AgentMessageInput(BaseModel):
    agent_handle: str
    message: str
    user_id: str = None
    conversation_id: str = None

class AgentRegisterInput(BaseModel):
    agent_handle: str
    api_key: str = None
    user_id: str = None
    conversation_id: str = None
    
class AgentItem(BaseModel):
    agent_handle: str
    description: str
    URL: str

class AgentPublishInput(BaseModel):
    agent: AgentItem = Field(..., example=[
        {"agent_handle": "@SuperDappAPI", "description": "SuperDapp API agent for send/recv crypto, messages and mails/files", "URL": "https://studio.superdapp.io/27728292"}])

class AgentOutput(BaseModel):
    """Model for agent output that excludes sensitive information."""
    handle: str
    description: str
    added_by: str | None = None
    added_at: datetime | None = None
    registered_at: datetime | None = None

    @classmethod
    def from_mongo(cls, mongo_doc: dict):
        """Create AgentOutput from MongoDB document, excluding sensitive fields."""
        return cls(
            handle=mongo_doc["handle"],
            description=mongo_doc["description"],
            added_by=mongo_doc.get("added_by"),
            added_at=mongo_doc.get("added_at"),
            registered_at=mongo_doc.get("registered_at")
        )

class AgentsManager:

    def __init__(self):
        load_dotenv()
        self.rate_limiter = RateLimiter(rate=10, period=1)
        self.init_lock = Lock()
        self.mongo_client = None
        self.db = None
        self.agents_collection = None
        self.conversation_agents_collection = None

    async def initialize(self):
        async with self.init_lock:
            if self.mongo_client is not None:
                return
            try:
                self.mongo_client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
                self.db = self.mongo_client.agents_db
                self.agents_collection = self.db.agents
                self.conversation_agents_collection = self.db.conversation_agents

                # Create indexes for agents collection
                await self.rate_limiter.execute(
                    self.agents_collection.create_index,
                    [("handle", 1)],
                    unique=True
                )
                await self.rate_limiter.execute(
                    self.agents_collection.create_index,
                    [("added_by", 1)]
                )
                await self.rate_limiter.execute(
                    self.agents_collection.create_index,
                    [("description", "text")]
                )

                # Create indexes for conversation_agents collection
                await self.rate_limiter.execute(
                    self.conversation_agents_collection.create_index,
                    [("conversation_id", 1), ("handle", 1)],
                    unique=True
                )
                await self.rate_limiter.execute(
                    self.conversation_agents_collection.create_index,
                    [("description", "text")]
                )

            except Exception as e:
                logging.warn(f"AgentsManager: initialize exception {e}\n{traceback.format_exc()}")

    async def list_registered_agents(self, agent_input: AgentListInput):
        """Fetch agents based on input criteria."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            if not agent_input.user_id and not agent_input.conversation_id:
                return "Error: user_id or conversation_id not provided", 0
            if agent_input.user_id and agent_input.conversation_id:
                return "Error: user_id and conversation_id both provided, can only provide one", 0

            response = await self.scan_agents(agent_input, 10)
            return response, time.time() - start

        except Exception as e:
            logging.warn(
                f"AgentsManager: listRegisteredAgents exception {e}\n{traceback.format_exc()}")
            return [], time.time() - start

    async def publish_agent(self, agent_input: AgentPublishInput):
        """Publish a new agent to the registry."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            logging.info("AgentsManager: publishing agent...")

            # Use rate limiter for MongoDB update
            await self.rate_limiter.execute(
                self.agents_collection.update_one,
                {"handle": agent_input.agent.agent_handle},
                {"$set": {
                    "description": agent_input.agent.description,
                    "URL": agent_input.agent.URL,
                    "added_at": datetime.utcnow()
                }},
                upsert=True
            )
            
        except Exception as e:
            logging.warn(f"AgentsManager: publish_agent exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(f"AgentsManager: publish_agent took {end - start} seconds")
            return agent_input.agent.name, end-start

    async def unpublish_agent(self, agent_handle: str):
        """Unpublish agent by removing it from the registry."""
        start = time.time()
        try:
            # Delete the agent and its conversation associations
            await self.delete_agent(agent_handle)
        except Exception as e:
            logging.error(
                f"AgentsManager: unpublish_agent failed, exception {e}\n{traceback.format_exc()}")
        return "Agent unpublished", time.time() - start

    async def delete_agent(self, agent_handle: str):
        """Delete agent from MongoDB and remove from all conversations."""
        if self.mongo_client is None:
            await self.initialize()
        start = time.time()
        try:
            # Delete the agent from agents collection with rate limiter
            result = await self.rate_limiter.execute(
                self.agents_collection.delete_one,
                {"handle": agent_handle}
            )
            
            # Remove agent from all conversations with rate limiter
            await self.rate_limiter.execute(
                self.conversation_agents_collection.delete_many,
                {"handle": agent_handle}
            )
            
            if result.deleted_count == 0:
                logging.warning(f"Agent {agent_handle} not found for deletion")
                return "Warning: Call was success but nothing was deleted", time.time() - start
            
        except Exception as e:
            logging.error(f"Failed to delete agent {agent_handle}: {str(e)}")
        return "Agent deleted", time.time() - start

    async def get_agent(self, agent_handle: str):
        """Retrieve a single agent by handle."""
        if self.mongo_client is None:
            await self.initialize()
            
        try:
            agent = await self.rate_limiter.execute(
                self.agents_collection.find_one,
                {"handle": agent_handle}
            )
            return agent if agent else None
            
        except Exception as e:
            logging.warn(f"AgentsManager: get_agent exception {e}\n{traceback.format_exc()}")
            return None

    async def add_agent_to_conversation(self, agent_input: AgentRegisterInput):
        """Add an agent to a conversation."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            if not agent_input.conversation_id:
                return "Error: conversation_id not provided", time.time() - start
            if not agent_input.agent_handle:
                return "Error: agent_handle not provided", time.time() - start

            # Check if agent exists
            agent = await self.get_agent(agent_input.agent_handle)
            if not agent:
                return "Error: Agent not found", time.time() - start

            await self.rate_limiter.execute(
                self.conversation_agents_collection.update_one,
                {
                    "conversation_id": agent_input.conversation_id,
                    "handle": agent_input.agent_handle
                },
                {"$set": {
                    "description": agent.get("description", "")
                }},
                upsert=True
            )
            
            return "Agent added to conversation", time.time() - start
            
        except Exception as e:
            logging.warn(f"AgentsManager: add_agent_to_conversation exception {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}", time.time() - start

    async def remove_agent_from_conversation(self, agent_input: AgentRegisterInput):
        """Remove an agent from a conversation."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            if not agent_input.conversation_id:
                return "Error: conversation_id not provided", time.time() - start
            if not agent_input.agent_handle:
                return "Error: agent_handle not provided", time.time() - start

            result = await self.rate_limiter.execute(
                self.conversation_agents_collection.delete_one,
                {
                    "conversation_id": agent_input.conversation_id,
                    "handle": agent_input.agent_handle 
                }
            )
            
            success = result.deleted_count > 0
            message = "Agent removed from conversation" if success else "Agent not found in conversation"
            return message, time.time() - start
            
        except Exception as e:
            logging.warn(f"AgentsManager: remove_agent_from_conversation exception {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}", time.time() - start

    async def scan_agents(self, agent_input: AgentListInput, page_size: int = 10):
        """Scan agents with pagination based on input criteria."""
        if self.mongo_client is None:
            await self.initialize()
            
        try:
            # Calculate skip for pagination
            skip = agent_input.results_page * page_size
            agents = []

            if agent_input.conversation_id:
                # Build query for conversation agents
                query = {"conversation_id": agent_input.conversation_id}
                if agent_input.filter:
                    query["$text"] = {"$search": agent_input.filter}
                elif agent_input.agent_handle:
                    query["handle"] = agent_input.agent_handle
                
                cursor = await self.rate_limiter.execute(
                    self.conversation_agents_collection.find,
                    query,
                    skip=skip,
                    limit=page_size
                )
                
                # For each conversation agent, lookup the full agent details
                async for convo_agent in cursor:
                    agent = await self.get_agent(convo_agent["handle"])
                    if agent:
                        agents.append(AgentOutput.from_mongo(agent))
            elif agent_input.user_id:
                # Direct query on agents collection
                query = {}
                if agent_input.user_id:
                    query["added_by"] = agent_input.user_id
                if agent_input.filter:
                    query["$text"] = {"$search": agent_input.filter}
                elif agent_input.agent_handle:
                    query["handle"] = agent_input.agent_handle
                    
                cursor = await self.rate_limiter.execute(
                    self.agents_collection.find,
                    query,
                    skip=skip,
                    limit=page_size
                )
                
                async for agent in cursor:
                    agents.append(AgentOutput.from_mongo(agent))
                    
            return agents
            
        except Exception as e:
            logging.warn(f"AgentsManager: scan_agents exception {e}\n{traceback.format_exc()}")
            return []

    async def register_agent(self, agent_input: AgentRegisterInput):
        """Register an agent for a user."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            if not agent_input.user_id:
                return "Error: user_id not provided", time.time() - start
            if not agent_input.api_key:
                return "Error: api_key not provided", time.time() - start

            # Check if agent exists
            agent = await self.get_agent(agent_input.agent_handle)
            if not agent:
                return "Error: agent not found", time.time() - start

            await self.rate_limiter.execute(
                self.agents_collection.update_one,
                {"handle": agent_input.agent_handle},
                {"$set": {
                    "api_key": agent_input.api_key,
                    "added_by": agent_input.user_id,
                    "registered_at": datetime.utcnow()
                }}
            )
            
            return "Agent registered successfully", time.time() - start
            
        except Exception as e:
            logging.warn(f"AgentsManager: register_agent exception {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}", time.time() - start

    async def message_agent(self, agent_input: AgentMessageInput):
        """Send message to an agent if authorized."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            if not agent_input.user_id and not agent_input.conversation_id:
                return "Error: user_id or conversation_id not provided", time.time() - start
            if not agent_input.message:
                return "Error: message not provided", time.time() - start
            if not agent_input.agent_handle:
                return "Error: agent_handle not provided", time.time() - start

            # Check if agent exists and get its details
            agent = await self.get_agent(agent_input.agent_handle)
            if not agent:
                return "Error: agent not found", time.time() - start
            
            # Check authorization
            is_authorized = False
            
            # Check if user is the agent's owner
            if agent.get("added_by") == agent_input.user_id:
                is_authorized = True
            
            # If not owner and conversation_id provided, check if agent is in conversation
            elif agent_input.conversation_id:
                convo_agent = await self.rate_limiter.execute(
                    self.conversation_agents_collection.find_one,
                    {
                        "conversation_id": agent_input.conversation_id,
                        "handle": agent_input.agent_handle
                    }
                )
                if convo_agent:
                    is_authorized = True
            
            if not is_authorized:
                return "Error: not authorized to message this agent", time.time() - start

            # Check if agent is registered (has API key)
            if not agent.get("api_key"):
                return "Error: agent is not registered", time.time() - start

            try:
                # Setup websocket connection to agent's URL
                async with websockets.connect(agent["URL"]) as websocket:
                    # Prepare message payload
                    payload = {
                        "agent": agent_input.agent_handle,
                        "message": agent_input.message,
                        "user_id": agent["added_by"],
                        "conversation_id": agent_input.conversation_id,
                        "api_key": agent["api_key"]
                    }
                    
                    # Send message
                    await websocket.send(json.dumps(payload))
                    
                    # Wait for response
                    response = await websocket.recv()
                    return json.loads(response), time.time() - start
                    
            except websockets.exceptions.WebSocketException as e:
                return f"Error connecting to agent: {str(e)}", time.time() - start
            
        except Exception as e:
            logging.warn(f"AgentsManager: message_agent exception {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}", time.time() - start
