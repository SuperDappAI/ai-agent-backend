import asyncio
import logging
from typing import Dict, List, Tuple, Union

import websockets
from fastapi import WebSocket, WebSocketDisconnect


class WebSocketConnectionManager:
    """
    Manages WebSocket connections including sending, broadcasting, and managing the lifecycle of connections.
    """

    def __init__(
        self,
        active_connections: List[Tuple[WebSocket, str]] = None,
        active_connections_lock: asyncio.Lock = None,
    ) -> None:
        """
        Initializes WebSocketConnectionManager with an optional list of active WebSocket connections.

        :param active_connections: A list of tuples, each containing a WebSocket object and its corresponding id.
        """
        if active_connections is None:
            active_connections = []
        self.active_connections_lock = active_connections_lock
        self.active_connections: List[Tuple[WebSocket, str]] = active_connections

    async def connect(self, websocket: WebSocket, id: str) -> None:
        """
        Accepts a new WebSocket connection and appends it to the active connections list.

        :param websocket: The WebSocket instance representing a client connection.
        :param id: A string representing the unique identifier of the client.
        """
        await websocket.accept()
        async with self.active_connections_lock:
            self.active_connections.append((websocket, id))
            print(f"New Client Connection: {id}, Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnects and removes a WebSocket connection from the active connections list.

        :param websocket: The WebSocket instance to remove.
        """
        async with self.active_connections_lock:
            try:
                self.active_connections = [conn for conn in self.active_connections if conn[0] != websocket]
                print(f"Client Connection Closed. Total: {len(self.active_connections)}")
            except ValueError:
                print("Error: WebSocket connection not found")

    async def disconnect_all(self) -> None:
        """
        Disconnects all active WebSocket connections.
        """
        for connection, _ in self.active_connections[:]:
            await self.disconnect(connection)
            
    async def send_message(self, message: Union[Dict, str], websocket: WebSocket) -> None:
        """
        Sends a JSON message to a single WebSocket connection.

        :param message: A JSON serializable dictionary containing the message to send.
        :param websocket: The WebSocket instance through which to send the message.
        """
        try:
            async with self.active_connections_lock:
                await websocket.send_json(message)
        except WebSocketDisconnect:
            print("Error: Tried to send a message to a closed WebSocket")
            await self.disconnect(websocket)
        except websockets.exceptions.ConnectionClosedOK:
            print("Error: WebSocket connection closed normally")
            await self.disconnect(websocket)
        except Exception as e:
            print(f"Error in sending message: {str(e)}", message)
            await self.disconnect(websocket)

    async def get_input(self, prompt: Union[Dict, str], websocket: WebSocket, timeout: int = 60) -> str:
        """
        Sends a JSON message to a single WebSocket connection as a prompt for user input.
        Waits on a user response or until the given timeout elapses.

        :param prompt: A JSON serializable dictionary containing the message to send.
        :param websocket: The WebSocket instance through which to send the message.
        """
        response = "Error: Unexpected response.\nTERMINATE"
        try:
            async with self.active_connections_lock:
                await websocket.send_json(prompt)
                result = await asyncio.wait_for(websocket.receive_json(), timeout=timeout)
                data = result.get("data")
                if data:
                    response = data.get("content", "Error: Unexpected response format\nTERMINATE")
                else:
                    response = "Error: Unexpected response format\nTERMINATE"

        except asyncio.TimeoutError:
            response = f"The user was timed out after {timeout} seconds of inactivity.\nTERMINATE"
        except WebSocketDisconnect:
            print("Error: Tried to send a message to a closed WebSocket")
            await self.disconnect(websocket)
            response = "The user was disconnected\nTERMINATE"
        except websockets.exceptions.ConnectionClosedOK:
            print("Error: WebSocket connection closed normally")
            await self.disconnect(websocket)
            response = "The user was disconnected\nTERMINATE"
        except Exception as e:
            print(f"Error in sending message: {str(e)}", prompt)
            await self.disconnect(websocket)
            response = f"Error: {e}\nTERMINATE"

        return response

    async def broadcast(self, message: Dict) -> None:
        """
        Broadcasts a JSON message to all active WebSocket connections.

        :param message: A JSON serializable dictionary containing the message to broadcast.
        """
        # Create a message dictionary with the desired format
        message_dict = {"message": message}

        for connection, _ in self.active_connections[:]:
            try:
                if connection.state == websockets.protocol.State.OPEN:
                    # Call send_message method with the message dictionary and current WebSocket connection
                    await self.send_message(message_dict, connection)
                else:
                    print("Error: WebSocket connection is closed")
                    await self.disconnect(connection)
            except (WebSocketDisconnect, websockets.exceptions.ConnectionClosedOK) as e:
                print(f"Error: WebSocket disconnected or closed({str(e)})")
                await self.disconnect(connection)
                

class MessageManager:
    """
    This class handles the automated generation and management of chat interactions
    using an automated workflow configuration and message queue.
    """

    def __init__(
        self, websocket_manager: WebSocketConnectionManager = None, human_input_timeout: int = 180
    ) -> None:
        """
        Initializes the AutoGenChatManager with a message queue.

        """
        self.websocket_manager = websocket_manager
        self.a_human_input_timeout = human_input_timeout

    async def a_send(self, message: dict) -> None:
        """
        Asynchronously sends a message via the WebSocketManager class

        :param message: The message string to be sent.
        """
        for connection, socket_client_id in self.websocket_manager.active_connections:
            if message["connection_id"] == socket_client_id:
                logging.info(
                    f"Sending message to client connection_id: {message['connection_id']}. Connection ID: {socket_client_id}"
                )
                await self.websocket_manager.send_message(message, connection)
            else:
                logging.info(
                    f"Skipping message for client connection_id: {message['connection_id']}. Connection ID: {socket_client_id}"
                )
    
    async def a_prompt_for_input(self, prompt: dict, timeout: int = 60) -> str:
        """
        Sends the user a prompt and waits for a response asynchronously via the WebSocketManager class

        :param message: The message string to be sent.
        """

        for connection, socket_client_id in self.websocket_manager.active_connections:
            if prompt["connection_id"] == socket_client_id:
                logging.info(
                    f"Sending message to client connection_id: {prompt['connection_id']}. Connection ID: {socket_client_id}"
                )
                try:
                    result = await self.websocket_manager.get_input(prompt, connection, timeout)
                    return result
                except Exception as e:
                    return f"Error: {e}\nTERMINATE"
            else:
                logging.info(
                    f"Skipping message for client connection_id: {prompt['connection_id']}. Connection ID: {socket_client_id}"
                )
