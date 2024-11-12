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
from websocket_connection_manager import MessageManager

class AgentListInput(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    results_page: Optional[int] = 0
    agent_handle: Optional[str] = None
    filter: Optional[str] = None

class ClearAgentMemory(BaseModel):
    agent_handle: str
    user_id: str
    conversation_id: str

class AgentMessageInput(BaseModel):
    agent_handle: str
    message: str
    user_id: str
    conversation_id: str
    context: str

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

class MessageMeta():
    task: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    summary_method: Optional[str] = "last"
    files: Optional[List[dict]] = None
    time: Optional[datetime] = None
    log: Optional[List[dict]] = None
    usage: Optional[List[dict]] = None

class Message():
    user_id: int
    role: str
    content: str
    session_id: int
    connection_id: str
    meta: Optional[Union[MessageMeta, dict]] = None

class AgentsManager:

    def __init__(self, message_manager: MessageManager):
        load_dotenv()
        self.rate_limiter = RateLimiter(rate=10, period=1)
        self.init_lock = Lock()
        self.mongo_client = None
        self.db = None
        self.agents_collection = None
        self.conversation_agents_collection = None
        self.sessions_collection = None
        self.max_description_length_allowed = 512
        self.message_manager = message_manager

    async def initialize(self):
        async with self.init_lock:
            if self.mongo_client is not None:
                return
            try:
                self.mongo_client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
                self.db = self.mongo_client.agents_db

                # Create collections if they do not exist
                self.agents_collection = self.db.get_collection("agents")
                self.conversation_agents_collection = self.db.get_collection("conversation_agents")
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
                        [("registered_to", "text"), ("description", "text")],
                        name="text_index"
                    )

                # Create a unique compound index for conversation_agents collection if it does not exist
                existing_indexes = await self.conversation_agents_collection.index_information()
                if "conversation_id_handle" not in existing_indexes:
                    await self.rate_limiter.execute(
                        self.conversation_agents_collection.create_index,
                        [("conversation_id", 1), ("handle", 1)],
                        unique=True
                    )

                # Create a single text index for conversation_agents collection if it does not exist
                if "text_index" not in existing_indexes:
                    await self.rate_limiter.execute(
                        self.conversation_agents_collection.create_index,
                        [("description", "text")],
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
                if "registered_to" not in existing_indexes:
                    await self.rate_limiter.execute(
                        self.registrations_collection.create_index,
                        [("registered_to", 1)]
                    )
                if "registered_to_handle" not in existing_indexes:
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
            workflow_url = f"{agent_input.URL}/api/workflows/{agent_input.workflow_id}?user_id={agent_input.user_id}"
            async with aiohttp.ClientSession() as session:
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
                return "Error: not authorized to unpublish this agent", time.time() - start

            # Find all sessions for this agent
            cursor = self.sessions_collection.find({"handle": agent_input.agent_handle})

            
            # Delete remote sessions
            async with aiohttp.ClientSession() as http_session:
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
            if agent_input.conversation_id:
                query = {"conversation_id": agent_input.conversation_id}
                if agent_input.filter:
                    query["$text"] = {"$search": agent_input.filter}
                elif agent_input.agent_handle:
                    query["handle"] = agent_input.agent_handle

                # Fetch conversation agents
                cursor = self.conversation_agents_collection.find(query).skip(skip).limit(page_size)
                handles = [convo_agent["handle"] async for convo_agent in cursor]

                # Fetch agents in bulk using `get_agents`
                if handles:
                    agent_docs = await self.get_agents(handles)
                    agents.extend(AgentOutput.from_mongo(agent, None) for agent in agent_docs)

            # If no agents found in conversation, look in registrations
            if len(agents) == 0:
                query = {"registered_to": agent_input.user_id}
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
                    agents.extend(AgentOutput.from_mongo(agent, agent_input.user_id) for agent in agent_docs)

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
            if agent.get("published_by") != agent_input.publisher_user_id:
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
            if agent.get("published_by") != agent_input.publisher_user_id:
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
        """Send message to an agent if authorized and stream responses."""
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
            # Check if user is in the agent's registration list (the registered user is calling the agent himself)
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
                # if user is not the registered agent owner must have registered a conversation and user is in the group conversation
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
            # if session doesn't exist locally, try to create a new one via API and store it
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
            
           # Connect to the agent and store the connection
            agent_ws_url = f"{agent.get('URL')}/api/ws/{agent_input.user_id}"
            try:
                async with websockets.connect(agent_ws_url) as agent_websocket:

                    # Prepare message content
                    message = f'CONTEXT: {agent_input.context}\nMESSAGE: {agent_input.message}' if agent_input.context else agent_input.message

                    # Create message payload
                    msg = Message(
                        user_id=registered_to,
                        role="user",
                        content=message,
                        session_id=session_id,
                        connection_id=agent_input.user_id
                    )
                    payload = json.dumps({
                        "type": "user_message",
                        "data": msg.dict()
                    })
                    
                    # Send message to the agent
                    await agent_websocket.send(payload)
                    
                    # Start the communication loop
                    while True:
                        # Receive message from the agent
                        agent_message = await agent_websocket.recv()
                        if not agent_message:
                            break  # Agent disconnected

                        # Parse agent_message from JSON
                        agent_message = json.loads(agent_message)

                        if agent_message.get('type') == 'user_input_request':
                            # Send prompt to client and wait for response
                            user_input = await self.message_manager.a_prompt_for_input(agent_message)
                            
                            # Prepare and send user's input back to the agent
                            message_payload = {
                                "recipient": agent_message.get("data").get("sender"),
                                "sender": agent_message.get("data").get("recipient"),
                                "message": user_input,
                                "timestamp": datetime.now().isoformat(),
                                "sender_type": agent_message.get("data").get("sender_type"),
                                "connection_id": agent_message.get("data").get("connection_id"),
                                "message_type": agent_message.get("data").get("message_type"),
                            }
                            socket_msg = json.dumps(
                                type="user_input_response",
                                data=message_payload,
                            )
                            await agent_websocket.send(socket_msg)
                        else:
                            # Forward message to client
                            await self.message_manager.a_send(agent_message)

                        if agent_message.get('type') == 'agent_response':
                            # Agent has finished processing
                            break
            except Exception as e:
                logging.error(f"Error during WebSocket communication: {str(e)}")
                await self.message_manager.a_send({'type': 'error', 'message': str(e)})
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
