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
import aiohttp


class AgentListInput(BaseModel):
    user_id: str
    conversation_id: str
    results_page: int = 0
    agent_handle: str = None
    filter: str = None
    
    def __str__(self):
        if self.filter:
            return self.user_id + self.conversation_id + self.filter + str(self.results_page)
        elif self.agent_handle:
            return self.user_id + self.conversation_id + self.agent_handle + str(self.results_page)
        else:
            return self.user_id + self.conversation_id + str(self.results_page)
    
    def __eq__(self, other):
        if self.filter:
            return self.user_id  == other.user_id and self.conversation_id  == other.conversation_id and self.filter == other.filter and self.results_page == other.results_page
        elif self.agent_handle:
            return self.user_id  == other.user_id and self.conversation_id  == other.conversation_id and self.agent_handle == other.agent_handle and self.results_page == other.results_page
        else:
            return self.user_id  == other.user_id and self.conversation_id  == other.conversation_id and self.results_page == other.results_page
 
    def __hash__(self):
        return hash(str(self))

class ClearAgentMemory(BaseModel):
    agent_handle: str
    user_id: str
    conversation_id: str

class AgentMessageInput(BaseModel):
    agent_handle: str
    message: str
    user_id: str
    conversation_id: str

class AgentRegisterInput(BaseModel):
    agent_handle: str
    user_id: str
    publisher_user_id: str
   
class AgentRegisterGroupInput(BaseModel):
    agent_handle: str
    user_id: str
    conversation_id: str

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
    registered_to: str

    @classmethod
    def from_mongo(cls, mongo_doc: dict, registered_to: str):
        """Create AgentOutput from MongoDB document, excluding sensitive fields."""
        return cls(
            handle=mongo_doc.get("handle"),
            description=mongo_doc.get("description"),
            published_by=mongo_doc.get("published_by"),
            published_at=mongo_doc.get("published_at"),
            registered_to=registered_to,
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
        self.sessions_collection = None
        self.max_description_length_allowed = 512

    async def initialize(self):
        async with self.init_lock:
            if self.mongo_client is not None:
                return
            try:
                self.mongo_client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
                self.db = self.mongo_client.agents_db
                self.agents_collection = self.db.agents
                self.conversation_agents_collection = self.db.conversation_agents
                self.sessions_collection = self.db.sessions
                self.registrations_collection = self.db.registrations

                # Create indexes for agents collection
                await self.rate_limiter.execute(
                    self.agents_collection.create_index,
                    [("handle", 1)],
                    unique=True
                )
                await self.rate_limiter.execute(
                    self.agents_collection.create_index,
                    [("registered_to", 1)]
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
                    [("handle", 1)]
                )
                await self.rate_limiter.execute(
                    self.conversation_agents_collection.create_index,
                    [("description", "text")]
                )

                # Create indexes for sessions collection
                await self.rate_limiter.execute(
                    self.sessions_collection.create_index,
                    [("conversation_id", 1), ("handle", 1)],
                    unique=True
                )
                await self.rate_limiter.execute(
                    self.sessions_collection.create_index,
                    [ ("handle", 1)]
                )
                # Create indexes for registrations collection
                await self.rate_limiter.execute(
                    self.registrations_collection.create_index,
                    [("registered_to", 1), ("handle", 1)],
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
            workflow_url = f"{agent.get('URL')}/api/workflows/{agent_input.workflow_id}?user_id=${agent_input.user_id}"

            async with aiohttp.ClientSession() as session:
                async with session.get(workflow_url) as response:
                    # Check if the request was successful
                    if response.status != 200:
                        raise ValueError("Workflow does not exist or could not be fetched.")

                    # Parse the JSON response
                    workflow = await response.json()

            # Get the description from the workflow
            description = workflow.get("description")
            # Limit the description to max_description_length_allowed characters
            if len(description) > self.max_description_length_allowed:
                description = description[:self.max_description_length_allowed] + "..."

            await self.rate_limiter.execute(
                self.agents_collection.update_one,
                {"handle": agent_input.agent.agent_handle},
                {"$set": {
                    "description": description,
                    "URL": agent_input.agent.URL,
                    "published_at": datetime.utcnow(),
                    "published_by": agent_input.user_id,
                    "workflow_id": agent_input.workflow_id
                }},
                upsert = True
            )
            
        except Exception as e:
            logging.warn(f"AgentsManager: publish_agent exception {e}\n{traceback.format_exc()}")
        finally:
            end = time.time()
            logging.info(f"AgentsManager: publish_agent took {end - start} seconds")
            return agent_input.agent.name, end-start

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
                return "Error: not authorized to unpublish this agent", time.time() - start

            # Find all sessions for this agent
            sessions = await self.rate_limiter.execute(
                self.sessions_collection.find,
                {"handle": agent_input.agent_handle}
            )
            
            # Delete remote sessions
            async with aiohttp.ClientSession() as http_session:
                async for session in sessions:
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
            
            # Remove agent from all conversations
            await self.rate_limiter.execute(
                self.conversation_agents_collection.delete_many,
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

    async def add_agent_to_conversation(self, agent_input: AgentRegisterGroupInput):
        """Add an agent to a conversation."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            # Check if agent exists
            agent = await self.get_agent(agent_input.agent_handle)
            if not agent:
                return "Error: Agent not found", time.time() - start
            # look up registrations to make sure user_id has been registered
            registration = await self.rate_limiter.execute(
                self.registrations_collection.find_one,
                {
                    "registered_to": agent_input.user_id,
                    "handle": agent_input.agent_handle
                }
            )
            if registration is None:
                return "Error: Agent not registered to this user", time.time() - start
  
            await self.rate_limiter.execute(
                self.conversation_agents_collection.update_one,
                {
                    "conversation_id": agent_input.conversation_id,
                    "handle": agent_input.agent_handle
                },
                {"$set": {
                    "description": agent.get("description", ""),
                    "registered_to": agent_input.user_id
                }},
                upsert = True
            )
            
            return "Agent added to conversation", time.time() - start
            
        except Exception as e:
            logging.warn(f"AgentsManager: add_agent_to_conversation exception {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}", time.time() - start

    async def remove_agent_from_conversation(self, agent_input: AgentRegisterGroupInput):
        """Remove an agent from a conversation."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            # look up registrations to make sure user_id has been registered
            registration = await self.rate_limiter.execute(
                self.registrations_collection.find_one,
                {
                    "registered_to": agent_input.user_id,
                    "handle": agent_input.agent_handle
                }
            )
            if registration is None:
                return "Error: Agent not registered to this user", time.time() - start

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
                    agents.append(AgentOutput.from_mongo(agent, convo_agent.get("registered_to")))
            # if conversation is not a group, look up on the user_id assuming the registrar is calling it
            if len(agents) == 0:
                # Direct query on agents collection
                query = {"registered_to": agent_input.user_id}
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
                    agents.append(AgentOutput.from_mongo(agent, agent_input.user_id))
                    
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
            # check publisher_user_id to make sure only he can register to other users
            if agent.published_by != agent_input.publisher_user_id:
                return "Error: not authorized to register this agent", time.time() - start

            await self.rate_limiter.execute(
                self.registrations_collection.update_one,
                {
                    "registered_to": agent_input.user_id,
                    "handle": agent_input.agent_handle
                },
                {"$set": {
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
            # check publisher_user_id to make sure only he can unregister
            if agent.published_by != agent_input.publisher_user_id:
                return "Error: not authorized to unregister this agent", time.time() - start

            await self.rate_limiter.execute(
                self.registrations_collection.delete_one,
                {
                    "registered_to": agent_input.user_id,
                    "handle": agent_input.agent_handle
                }
            )
            
            return "Agent unregistered successfully", time.time() - start
            
        except Exception as e:
            logging.warn(f"AgentsManager: unregister_agent exception {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}", time.time() - start

    async def message_agent(self, agent_input: AgentMessageInput):
        """Send message to an agent if authorized."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            # Check if agent exists and get its details
            agent = await self.get_agent(agent_input.agent_handle)
            if not agent:
                return "Error: agent not found", time.time() - start
            
            # Check authorization
            is_authorized = False
            registered_to = None
            # Check if user is in the agent's registration list
            registration = await self.rate_limiter.execute(
                self.registrations_collection.find_one,
                {
                    "registered_to": agent_input.user_id,
                    "handle": agent_input.agent_handle
                }
            )
            if registration is not None:
                registered_to = agent_input.user_id
                is_authorized = True
            else:
                # if user is not the registered agent owner must have registered a conversation and user is in it
                convo_agent = await self.rate_limiter.execute(
                    self.conversation_agents_collection.find_one,
                    {
                        "conversation_id": agent_input.conversation_id,
                        "handle": agent_input.agent_handle
                    }
                )
                if convo_agent is not None:
                    registered_to = convo_agent.get('registered_to')
                    is_authorized = True
            if not is_authorized:
                return "Error: not authorized to message this agent", time.time() - start

            # Check if session exists for this conversation
            session = None
            if agent_input.conversation_id:
                session = await self.rate_limiter.execute(
                    self.sessions_collection.find_one,
                    {
                        "conversation_id": agent_input.conversation_id,
                        "handle": agent_input.agent_handle
                    }
                )

            if session is None:
                # Create new session via REST POST
                async with aiohttp.ClientSession() as http_session:
                    session_data = {
                        "user_id": registered_to,
                        "workflow_id": agent.get("workflow_id")
                    }
                    async with http_session.post(f"{agent.get('URL')}/api/sessions", json=session_data) as response:
                        if response.status != 200:
                            return "Error: Failed to create agent session", time.time() - start
                        session_response = await response.json()
                        session_id = session_response["data"]["id"]
                        
                        # Store session info
                        await self.rate_limiter.execute(
                            self.sessions_collection.insert_one,
                            {
                                "conversation_id": agent_input.conversation_id,
                                "handle": agent_input.agent_handle,
                                "session_id": session_id,
                                "registered_to": registered_to,
                            }
                        )
            else:
                session_id = session.get("session_id")

            try:
                # Setup websocket connection to agent's URL
                async with websockets.connect(agent.get("URL") + "/api/ws/" + str(session_id)) as websocket:
                    # Prepare message payload
                    payload = {
                        "type": "user_message",
                        "data": agent_input.message,
                        "workflow_id": agent.workflow_id,
                        "registered_to": registered_to,
                        "session_id": session_id
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

    async def clear_conversation(self, clear_memory: ClearAgentMemory):
        """Send clear agent conversation."""
        if self.mongo_client is None:
            await self.initialize()
            
        start = time.time()
        try:
            # Lookup agent
            agent = await self.get_agent(clear_memory.agent_handle)
            if not agent:
                return "Error: agent not found", time.time() - start

            # Check authorization
            if agent.get("registered_to") != clear_memory.user_id:
                return "Error: not authorized to clear this agent's conversation", time.time() - start

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

            # Delete session via REST call
            async with aiohttp.ClientSession() as http_session:
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
