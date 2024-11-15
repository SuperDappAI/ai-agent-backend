import time
import logging
import traceback
from dotenv import load_dotenv
import os
from datetime import datetime
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from asyncio import Lock
from rate_limiter import RateLimiter
import json
import websockets
import aiohttp
from typing import Any, Dict, List, Optional, Union

class AgentListInput(BaseModel):
    user_id: str
    results_page: Optional[int] = 0
    agent_handle: Optional[str] = None
    filter: Optional[str] = None

class ClearAgentMemory(BaseModel):
    agent_handle: str
    user_id: str
    conversation_id: str

class AgentMessageInput(BaseModel):
    agent_handle: str
    user_id: str
    conversation_id: str

class AgentMessageOutput(BaseModel):
    status: str
    URL: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    api_key: Optional[str] = None

class AgentRegisterInput(BaseModel):
    agent_handle: str
    user_id: str
    api_key: Optional[str] = None
   
class AgentPublishInput(BaseModel):
    user_id: str
    workflow_id: str
    agent_handle: str
    URL: str

class AgentUnpublishInput(BaseModel):
    agent_handle: str
    user_id: str

class AgentOutput(BaseModel):
    """Model for agent output that excludes sensitive information."""
    handle: str
    description: str
    published_by: str
    published_at: datetime

    @classmethod
    def from_mongo(cls, mongo_doc: dict):
        """Create AgentOutput from MongoDB document, excluding sensitive fields."""
        return cls(
            handle=mongo_doc.get("handle"),
            description=mongo_doc.get("description"),
            published_by=mongo_doc.get("published_by"),
            published_at=mongo_doc.get("published_at"),
        )


class AgentsManager:

    def __init__(self):
        load_dotenv()
        self.rate_limiter = RateLimiter(rate=10, period=1)
        self.init_lock = Lock()
        self.mongo_client = None
        self.db = None
        self.agents_collection = None
        self.sessions_collection = None
        self.registrations_collection = None
        self.max_description_length_allowed = 512

    async def initialize(self):
        async with self.init_lock:
            if self.mongo_client is not None:
                return
            try:
                self.mongo_client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
                self.db = self.mongo_client.agents_db

                # Create collections if they do not exist
                self.agents_collection = self.db.get_collection("agents")
                self.sessions_collection = self.db.get_collection("sessions")
                self.registrations_collection = self.db.get_collection("registrations")

                # Create a unique index for handle in agents collection if it does not exist
                existing_indexes = await self.agents_collection.index_information()
                if "handle_1" not in existing_indexes:
                    await self.rate_limiter.execute(
                        self.agents_collection.create_index,
                        [("handle", 1)],
                        unique=True
                    )

                # Create a single text index for agents collection if it does not exist
                if "text_index" not in existing_indexes:
                    await self.rate_limiter.execute(
                        self.agents_collection.create_index,
                        [("user_id", "text"), ("description", "text")],
                        name="text_index"
                    )

                # Create a compound index for sessions collection if it does not exist
                existing_indexes = await self.sessions_collection.index_information()
                if "conversation_id_handle" not in existing_indexes:
                    await self.rate_limiter.execute(
                        self.sessions_collection.create_index,
                        [("conversation_id", 1), ("handle", 1)],
                        unique=True
                    )

                # Create a compound index for registrations collection if it does not exist
                existing_indexes = await self.registrations_collection.index_information()
                if "user_id" not in existing_indexes:
                    await self.rate_limiter.execute(
                        self.registrations_collection.create_index,
                        [("user_id", 1)]
                    )
                if "user_id_handle" not in existing_indexes:
                    await self.rate_limiter.execute(
                        self.registrations_collection.create_index,
                        [("user_id", 1), ("handle", 1)],
                        unique=True
                    )
            except Exception as e:
                logging.warn(f"AgentsManager: initialize exception {e}\n{traceback.format_exc()}")

    async def list_registered_agents(self, agent_input: AgentListInput):
        """Fetch agents based on input criteria."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
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
            workflow_url = f"{agent_input.URL}/api/workflows/{agent_input.workflow_id}?user_id={agent_input.user_id}"
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:
                async with session.get(workflow_url) as response:
                    # Check if the request was successful
                    if response.status != 200:
                        return "Workflow does not exist or could not be fetched.", 0

                    # Parse the JSON response
                    workflow = await response.json()
            if 'message' not in workflow["message"] != 'Workflow Retrieved Successfully' or 'status' not in workflow or workflow["status"] != True:
                return f"Could not retrieve workflow successfully via: {workflow_url}", 0
            if 'data' not in workflow or len(workflow["data"]) == 0:
                return f"Workflow {agent_input.workflow_id} does not exist.", 0
            if len(workflow["data"]) > 1:
                logging.warn(f"Workflow found multiple workflows with ID: {agent_input.workflow_id}.")
            # Get the description from the workflow
            description = workflow["data"][0]["description"]
            # Limit the description to max_description_length_allowed characters
            if len(description) > self.max_description_length_allowed:
                description = description[:self.max_description_length_allowed] + "..."

            await self.rate_limiter.execute(
                self.agents_collection.update_one,
                {"handle": agent_input.agent_handle},
                {"$set": {
                    "description": description,
                    "URL": agent_input.URL,
                    "published_at": datetime.utcnow(),
                    "published_by": agent_input.user_id,
                    "workflow_id": agent_input.workflow_id
                }},
                upsert = True
            )
            
        except Exception as e:
            return f"Workflow exception {e}", 0
  
        end = time.time()
        logging.info(f"AgentsManager: publish_agent took {end - start} seconds")
        return f"Succesffully published: {agent_input.agent_handle}", end-start

    async def unpublish_agent(self, agent_input: AgentUnpublishInput):
        """Unpublish agent from MongoDB and remove from all conversations."""
        if self.mongo_client is None:
            await self.initialize()
        start = time.time()
        try:
            # Get agent details first (we need the URL for session cleanup)
            agent = await self.get_agent(agent_input.agent_handle)
            if not agent:
                return "Error: Agent does not exist", time.time() - start
            # Check authorization
            if agent.get("published_by") != agent_input.user_id:
                print(f'agent.get("published_by") {agent.get("published_by")} agent_input.user_id {agent_input.user_id}')
                return "Error: not authorized to unpublish this agent", time.time() - start

            # Find all sessions for this agent
            cursor = self.sessions_collection.find({"handle": agent_input.agent_handle})

            # Delete remote sessions
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as http_session:
                async for session in cursor:
                    try:
                        delete_url = f"{agent.get('URL')}/api/sessions/delete?session_id={session.get('session_id')}&user_id={session.get('user_id')}"
                        async with http_session.delete(delete_url) as response:
                            if response.status != 200:
                                logging.warning(f"Failed to delete remote session {session.get('session_id')} for agent {agent_input.agent_handle}")
                    except Exception as e:
                        logging.warning(f"Error deleting remote session: {str(e)}")

            # Delete all local sessions for this agent
            await self.rate_limiter.execute(
                self.sessions_collection.delete_many,
                {"handle": agent_input.agent_handle}
            )
            # Delete all registrations for this agent
            await self.rate_limiter.execute(
                self.registrations_collection.delete_many,
                {"handle": agent_input.agent_handle}
            )
           
            # Delete the agent from agents collection
            result = await self.rate_limiter.execute(
                self.agents_collection.delete_one,
                {"handle": agent_input.agent_handle}
            )
            
            if result.deleted_count == 0:
                logging.warning(f"Agent {agent_input.agent_handle} not found for deletion")
                return "Warning: Call was success but nothing was deleted", time.time() - start
            
        except Exception as e:
            logging.error(f"Failed to unpublish agent {agent_input.agent_handle}: {str(e)}")
        return "Agent unpublished", time.time() - start

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

    async def get_agents(self, agent_handles: List[str]) -> List[Dict]:
        """Retrieve multiple agents by their handles."""
        if self.mongo_client is None:
            await self.initialize()

        try:
            cursor = self.agents_collection.find({"handle": {"$in": agent_handles}})
            agents = [agent async for agent in cursor]
            return agents

        except Exception as e:
            logging.warn(f"AgentsManager: get_agents exception {e}\n{traceback.format_exc()}")
            return []

    async def scan_agents(self, agent_input: AgentListInput, page_size: int = 10):
        """Scan agents with pagination based on input criteria."""
        if self.mongo_client is None:
            await self.initialize()

        try:
            # Calculate skip for pagination
            skip = agent_input.results_page * page_size
            agents = []

            query = {"user_id": agent_input.user_id}
            if agent_input.filter:
                query["$text"] = {"$search": agent_input.filter}
            elif agent_input.agent_handle:
                query["handle"] = agent_input.agent_handle

            # Fetch registered agents
            cursor = self.registrations_collection.find(query).skip(skip).limit(page_size)
            handles = [registered_agent["handle"] async for registered_agent in cursor]

            # Fetch agents in bulk using `get_agents`
            if handles:
                agent_docs = await self.get_agents(handles)
                agents.extend(AgentOutput.from_mongo(agent) for agent in agent_docs)

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
            # Check if agent exists
            agent = await self.get_agent(agent_input.agent_handle)
            if not agent:
                return "Error: agent not found", time.time() - start
            # TODO: check api_key against agent URL to see if its valid
            # pass empty api key to unregister it, otherwise register
            if agent_input.api_key is None:
                self.unregister_agent(agent_input)
            else:
                await self.rate_limiter.execute(
                    self.registrations_collection.update_one,
                    {
                        "user_id": agent_input.user_id,
                        "handle": agent_input.agent_handle
                    },
                    {"$set": {
                        "api_key": agent_input.api_key
                    }},
                    upsert=True
                )
                        
            return "Agent registered successfully", time.time() - start
            
        except Exception as e:
            logging.warn(f"AgentsManager: register_agent exception {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}", time.time() - start

    async def unregister_agent(self, agent_input: AgentRegisterInput):
        """Unregister an agent for a user."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            # Check if agent exists
            agent = await self.get_agent(agent_input.agent_handle)
            if not agent:
                return "Error: agent not found", time.time() - start

            await self.rate_limiter.execute(
                self.registrations_collection.delete_one,
                {
                    "user_id": agent_input.user_id,
                    "handle": agent_input.agent_handle
                }
            )
            
            return "Agent unregistered successfully", time.time() - start
            
        except Exception as e:
            logging.warn(f"AgentsManager: unregister_agent exception {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}", time.time() - start

    async def message_agent(self, agent_input: AgentMessageInput) -> AgentMessageOutput:
        """Send message to an agent if authorized and stream responses."""
        if self.mongo_client is None:
            await self.initialize()
            
        try:
            # Check if agent exists and get its details
            agent = await self.get_agent(agent_input.agent_handle)
            if not agent:
                return AgentMessageOutput(
                    status="Error: agent not found"
                )
            
            # Check authorization
            is_authorized = False
            # Check if user is in the agent's registration list (the registered user is calling the agent himself)
            registration = await self.rate_limiter.execute(
                self.registrations_collection.find_one,
                {
                    "user_id": agent_input.user_id,
                    "handle": agent_input.agent_handle
                }
            )
            if registration is None:
                return AgentMessageOutput(
                    status="Error: agent not authorized"
                )

            # Check if session exists for this conversation
            session = await self.rate_limiter.execute(
                self.sessions_collection.find_one,
                {
                    "conversation_id": agent_input.conversation_id,
                    "handle": agent_input.agent_handle
                }
            )
            # if session doesn't exist locally, try to create a new one via API and store it
            if session is None:
                # Create new session via REST POST
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as http_session:
                    session_data = {
                        "user_id": agent_input.user_id,
                        "workflow_id": agent.get("workflow_id")
                    }
                    async with http_session.post(f"{agent.get('URL')}/api/sessions", json=session_data) as response:
                        if response.status != 200:
                            return AgentMessageOutput(
                                status="Error: Failed to create agent session"
                            )
                        session_response = await response.json()
                        session_id = session_response["data"]["id"]
                        # Store session info
                        await self.rate_limiter.execute(
                            self.sessions_collection.insert_one,
                            {
                                "conversation_id": agent_input.conversation_id,
                                "handle": agent_input.agent_handle,
                                "session_id": session_id,
                                "user_id": agent_input.user_id,
                            }
                        )
            else:
                session_id = session.get("session_id")

        except Exception as e:
            logging.warn(f"AgentsManager: message_agent exception {e}\n{traceback.format_exc()}")
            return AgentMessageOutput(
                    status=f"Error: {str(e)}", 
            )
        # Prepare message content
        return AgentMessageOutput(
                status="Success",
                URL=agent.get("URL"),
                session_id=str(session_id),
                workflow_id=agent.get("workflow_id"),
                api_key=registration.get("api_key")
        )

    async def clear_conversation(self, clear_memory: ClearAgentMemory):
        """Send clear agent conversation."""
        if self.mongo_client is None:
            await self.initialize()

        start = time.time()
        try:
            # Lookup session
            session = await self.rate_limiter.execute(
                self.sessions_collection.find_one,
                {
                    "conversation_id": clear_memory.conversation_id,
                    "handle": clear_memory.agent_handle
                }
            )
            
            if session is None:
                return "Error: no session found for this conversation", time.time() - start
            
            # Check authorization
            if session.get("user_id") != clear_memory.user_id:
                return "Error: not authorized to clear this agent's conversation", time.time() - start

            # Delete session via REST call
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as http_session:
                delete_url = f"{agent.get('URL')}/api/sessions/delete?session_id={session.get('session_id')}&user_id={clear_memory.user_id}"
                async with http_session.delete(delete_url) as response:
                    if response.status != 200:
                        return "Error: Failed to clear agent session", time.time() - start

            # Delete local session record
            await self.rate_limiter.execute(
                self.sessions_collection.delete_one,
                {
                    "conversation_id": clear_memory.conversation_id,
                    "handle": clear_memory.agent_handle
                }
            )

            return "Conversation cleared successfully", time.time() - start

        except Exception as e:
            logging.warn(f"AgentsManager: clear_conversation exception {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}", time.time() - start
