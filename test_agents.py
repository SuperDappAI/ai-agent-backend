import asyncio
import websockets
import aiohttp
import json
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Define the base URL for the API
BASE_URL = "http://localhost:8111"

class MessageMeta(BaseModel):
    task: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    summary_method: Optional[str] = "last"
    files: Optional[List[dict]] = None
    time: Optional[datetime] = None
    log: Optional[List[dict]] = None
    usage: Optional[List[dict]] = None

class Message(BaseModel):
    user_id: str
    role: str
    content: str
    session_id: str
    workflow_id: Optional[str] = None # sent along to agent server but not received
    connection_id: str
    api_key: Optional[str] = None # sent along to agent server but not received
    meta: Optional[Union[dict, "MessageMeta"]] = None
    
    
async def publish_agent(session, agent_handle, user_id, workflow_id, url):
    """Publish an agent."""
    payload = {
        "agent_handle": agent_handle,
        "user_id": user_id,
        "workflow_id": workflow_id,
        "URL": url,
    }
    async with session.post(f"{BASE_URL}/publish_agent/", json=payload) as response:
        result = await response.json()
        print("Publish Agent Response:", result)

async def register_agent(session, agent_handle, user_id, api_key):
    """Register an agent."""
    payload = {
        "agent_handle": agent_handle,
        "user_id": user_id,
        "api_key": api_key
    }
    async with session.post(f"{BASE_URL}/register_agent/", json=payload) as response:
        result = await response.json()
        print("Register Agent Response:", result)

def wsFromURL(URL: str, user_id: str) -> str:
    agent_http_url = URL
    if agent_http_url.startswith("http://"):
        agent_ws_url = agent_http_url.replace("http://", "ws://", 1)
    elif agent_http_url.startswith("https://"):
        agent_ws_url = agent_http_url.replace("https://", "wss://", 1)
    else:
        raise ValueError(f"Invalid agent URL: {agent_http_url}")

    return f"{agent_ws_url}/api/ws/{user_id}"

async def send_message_via_websocket(session, agent_handle, user_id, conversation_id, message, context=None):
    """Send a message to an agent using WebSocket."""
    payload = {
        "agent_handle": agent_handle,
        "user_id": user_id,
        "conversation_id": conversation_id,
    }
    async with session.post(f"{BASE_URL}/message_agent/", json=payload) as response:
        result = await response.json()
        if result.get("status") != "Success":
            print(f"Error preparing message: {result.get('status')}")
            return
    
        print("Register Agent Response:", result)
        message = f'CONTEXT: {context}\nMESSAGE: {message}' if context else message
        msg = Message(
            user_id=user_id,
            role="user",
            content=message,
            session_id=result.get('session_id'),
            workflow_id=result.get("workflow_id"),
            connection_id=user_id
        )
        async with websockets.connect(wsFromURL(result.get("URL"), user_id)) as websocket:
            payload = json.dumps({
                "type": "user_message",
                "data": msg.dict()
            })
            await websocket.send(payload)
            print("Message Sent:", json.dumps(payload, indent=4))
            
            while True:
                try:
                    response = await websocket.recv()
                    print("Received:", response)
                    response_data = json.loads(response)
                    if response_data.get("type") == "agent_response":
                        print("Agent response received. Exiting...")
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed.")
                    break
            await websocket.close()

async def unregister_agent(session, agent_handle, user_id):
    """Unregister an agent."""
    payload = {
        "agent_handle": agent_handle,
        "user_id": user_id,
    }
    async with session.post(f"{BASE_URL}/unregister_agent/", json=payload) as response:
        result = await response.json()
        print("Unregister Agent Response:", result)

async def unpublish_agent(session, agent_handle, user_id):
    """Unpublish an agent."""
    payload = {
        "agent_handle": agent_handle,
        "user_id": user_id,
    }
    async with session.post(f"{BASE_URL}/unpublish_agent/", json=payload) as response:
        result = await response.json()
        print("Unpublish Agent Response:", result)

async def list_registered_agents(session, user_id, conversation_id=None, results_page=0, agent_handle=None, filter_text=None):
    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "results_page": results_page,
        "agent_handle": agent_handle,
        "filter": filter_text,
    }
    async with session.post(f"{BASE_URL}/list_registered_agents/", json=payload) as response:
        result = await response.json()
        print(f"List Registered Agents Response: {json.dumps(result, indent=4)}")


async def main():
    """Main function to handle command-line interactions."""
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:
        while True:
            print("\nAvailable Actions:")
            print("1. Publish Agent")
            print("2. Register Agent")
            print("3. Unregister Agent")
            print("4. Send Message to Agent")
            print("5. List Registered Agents")
            print("6. Exit")
            choice = input("Enter your choice: ")

            if choice == "1":
                agent_handle = input("Agent Handle: ")
                user_id = input("User ID: ")
                workflow_id = input("Workflow ID: ")
                url = input("Agent URL: ")
                await publish_agent(session, agent_handle, user_id, workflow_id, url)

            elif choice == "2":
                agent_handle = input("Agent Handle: ")
                user_id = input("Register to User ID: ")
                api_key = input("API KEY: ")
                await register_agent(session, agent_handle, user_id, api_key)

            elif choice == "3":
                agent_handle = input("Agent Handle: ")
                user_id = input("Unregistering User ID: ")
                await unregister_agent(session, agent_handle, user_id)

            elif choice == "4":
                agent_handle = input("Agent Handle: ")
                user_id = input("User ID: ")
                conversation_id = input("Conversation ID: ")
                message = input("Message: ")
                context = input("Context (optional): ") or None
                await send_message_via_websocket(session, agent_handle, user_id, conversation_id, message, context)

            elif choice == "5":
                user_id = input("User ID: ")
                conversation_id = input("Conversation ID (optional): ") or None
                results_page = int(input("Results Page (default 0): ") or 0)
                agent_handle = input("Agent Handle (optional): ") or None
                filter_text = input("Filter (optional): ") or None
                await list_registered_agents(session, user_id, conversation_id, results_page, agent_handle, filter_text)

            elif choice == "6":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())
