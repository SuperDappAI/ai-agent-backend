import asyncio
import websockets
import aiohttp
import json

# Define the base URL for the API
BASE_URL = "http://localhost:8000"

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

async def register_agent(session, agent_handle, user_id, publisher_user_id):
    """Register an agent."""
    payload = {
        "agent_handle": agent_handle,
        "user_id": user_id,
        "publisher_user_id": publisher_user_id,
    }
    async with session.post(f"{BASE_URL}/register_agent/", json=payload) as response:
        result = await response.json()
        print("Register Agent Response:", result)

async def send_message_via_websocket(agent_handle, user_id, conversation_id, message, context=None):
    """Send a message to an agent using WebSocket."""
    websocket_url = f"ws://localhost:8000/ws/{user_id}"  # Update if needed
    async with websockets.connect(websocket_url) as websocket:
        payload = {
            "agent_handle": agent_handle,
            "message": message,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "context": context,
        }
        await websocket.send(json.dumps(payload))
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

async def unregister_agent(session, agent_handle, user_id, publisher_user_id):
    """Unregister an agent."""
    payload = {
        "agent_handle": agent_handle,
        "user_id": user_id,
        "publisher_user_id": publisher_user_id,
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

async def main():
    """Main function to handle command-line interactions."""
    async with aiohttp.ClientSession() as session:
        while True:
            print("\nAvailable Actions:")
            print("1. Publish Agent")
            print("2. Register Agent")
            print("3. Send Message to Agent")
            print("4. Unregister Agent")
            print("5. Unpublish Agent")
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
                user_id = input("User ID: ")
                publisher_user_id = input("Publisher User ID: ")
                await register_agent(session, agent_handle, user_id, publisher_user_id)

            elif choice == "3":
                agent_handle = input("Agent Handle: ")
                user_id = input("User ID: ")
                conversation_id = input("Conversation ID: ")
                message = input("Message: ")
                context = input("Context (optional): ") or None
                await send_message_via_websocket(agent_handle, user_id, conversation_id, message, context)

            elif choice == "4":
                agent_handle = input("Agent Handle: ")
                user_id = input("User ID: ")
                publisher_user_id = input("Publisher User ID: ")
                await unregister_agent(session, agent_handle, user_id, publisher_user_id)

            elif choice == "5":
                agent_handle = input("Agent Handle: ")
                user_id = input("User ID: ")
                await unpublish_agent(session, agent_handle, user_id)

            elif choice == "6":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())
