import asyncio
import pytest
from unittest import mock
from agent_manager import AgentManager, MemoryInput, MemoryOutput, ClearMemory

@pytest.fixture
def agent_manager():
    agent_manager = AgentManager()

    # Mock time-consuming or IO-bound functions
    agent_manager.create_new_memory_retriever = mock.Mock()
    agent_manager.memory = mock.Mock()
    return agent_manager

@pytest.mark.asyncio
async def test_push_memory(agent_manager):
    test_memory_output = MemoryOutput(user_id='1', query='Test query', llm_response='Test response', conversation_id='1', importance='high')
    duration = await agent_manager.push_memory(test_memory_output)
    # Check that the save_context function was called once with the correct argument
    agent_manager.memory.save_context.assert_called_once_with(test_memory_output.dict())
    assert duration >= 0  # Ensures it took some time

def test_pull_memory(agent_manager):
    test_memory_input = MemoryInput(user_id='1', query='Test query', conversation_id='1')
    response, duration = agent_manager.pull_memory(test_memory_input)
    # make sure load_memory_variables is called
    agent_manager.memory.load_memory_variables.assert_called_once_with(queries=[test_memory_input.query], conversation_id=test_memory_input.conversation_id)
    assert duration >= 0  # Check that the operation took some time

def test_clear_conversation(agent_manager):
    test_clear_memory = ClearMemory(user_id='1', conversation_id='1')
    response, duration = agent_manager.clear_conversation(test_clear_memory)
    agent_manager.memory.clear.assert_called_once_with(test_clear_memory.conversation_id)  # make sure clear method is called
    assert duration >= 0  # Check that the operation took some time
    assert response == "success"  # Make sure response is success


