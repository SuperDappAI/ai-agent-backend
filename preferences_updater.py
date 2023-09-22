
import traceback
import logging
import json
import re

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from preferences_resolver import PreferencesResolver
from typing import List

class SystemPrompt:
    def __init__(self, preferences: str, error: str = ""):
         self.preferences = preferences
         self.error = error

    def to_prompt_string(self) -> str:
        template_text = f"""
        You are an expert preferences interpreter. Your task is to infer and update user preferences based on the dialog between the user and AI. Use the existing PREFERENCES as the current MongoDB database state.
        
        Guidelines:
        1. Utilize JsonPatch operations (OPS) for DB updates:
        - 'add': '/traits/-'
        - 'remove' or 'replace': '/traits/[index]'
        2. Update based on user/AI dialog, ensuring new preferences are unique, case-insensitive, and meaningful for future interactions.
        
        {self.error}
        
        Notes:
        - Only perform operations for meaningful state changes.
        - Use '-' with 'add' for appending to arrays.
        - Indices start at 0.
        - Adjust the DB schema as needed.
        - Only 'replace' for specified index changes.
        - Manage tasks/subtasks and ensure IDs cross-reference.
        - Cross-reference IDs of active tasks with tasks and IDs of active subtasks with subtasks.
        - If the LLM response indicates moving on to the next subtask or task, assume the current subtask is complete and advance sequentially. For milestone tasks, add to accomplishments upon completion.
        - Do not revert to a previous subtask within a task or previous task unless explicitly indicated by the user or LLM response.
        - In cases of ambiguity or confusion, default to no changes to prevent unintended switches.
        - Follow JsonPatch formatting rigorously.
        - Returning an empty list is acceptable.

        Consider token limits (max 1000) for the database. Ensure any updates won't exceed this. Prioritize new preferences over older ones if needed.
        
        Example OPS formats:
        OPS: []
        OPS: [{{"op": "add", "path": "/traits/-", "value": "adventurous"}},{{"op": "remove", "path": "/traits/1"}},{{"op": "replace", "path": "/traits/0", "value": "meticulous"}}]
        OPS: [{{"op": "add", "path": "/tasks/-", "value": {{"id": "task_0", "description": "New Task"}}}},{{"op": "add", "path": "/subtasks/-", "value": {{"id": "subtask_0", "task_id": "task_0", "description": "New Subtask"}}}},{{"op": "replace", "path": "/active_task_id", "value": "task_0"}},{{"op": "replace", "path": "/active_subtask_id", "value": "subtask_0"}}]
        OPS: [{{"op": "add", "path": "/achievements/-", "value": "New Achievement"}},{{"op": "add", "path": "/skills/-", "value": "New Skill"}},{{"op": "replace", "path": "/mood_feelings/0", "value": "content"}}]

        PREFERENCES: {self.preferences}
        """
        return template_text

class PreferencesUpdater:
    _preferences_resolver: PreferencesResolver
    _verbose: bool

    def __init__(self, preferences_resolver: PreferencesResolver, verbose: bool = False) -> None:
        self._preferences_resolver = preferences_resolver
        self._verbose = verbose

    async def _get_json_patch_commands(
        self, messages, llm: ChatOpenAI
    ) -> List[str]:
        """Generate 'preference updates', based on pertinent memories."""
        array_json = []
        try:
            response = await llm.agenerate(messages)
            if not response.generations or not response.generations[0]:
                raise Exception("LLM did not provide a valid summary response.")

            result = response.generations[0][0].text
            # Find the array in the output string using a regular expression
            array_match = re.search(r'OPS:\s*\[\s*(\{.*\})\s*\]', result, re.DOTALL)
            if array_match:
                array_str = '[' + array_match.group(1) + ']'
                array_str = array_str.replace("True", "true").replace("False", "false")
                # Parse the array string as JSON
                array_json = json.loads(array_str)
        except Exception as e:
            if self._verbose:
                logging.warn(f"PreferencesUpdater: _get_json_patch_commands exception, e: {e}\n{traceback.format_exc()}")

        return array_json

    async def update_preferences(self, llm: ChatOpenAI, user: str, ai: str, user_id: str):
        """Reflect on recent observations and generate 'insights'."""
        doc = await self._preferences_resolver.get_preferences(user_id)
        if doc is None:
            doc = self._preferences_resolver.default_preferences
        summary_prompt = SystemPrompt(doc)
        messages = [[SystemMessage(content=summary_prompt.to_prompt_string()), 
                    HumanMessage(content=user),
                    AIMessage(content=ai)]]
        patch_commands = await self._get_json_patch_commands(messages, llm)
        if len(patch_commands) > 0:
            if self._verbose:
                logging.info("AiDA is trying to update preferences")
            response = await self._preferences_resolver.apply_patch(user_id, doc, patch_commands)
            if response != "success":
                summary_prompt = SystemPrompt(doc, "2. You have been given human feedback that your changes were not accepted due to syntax, you are to carefully analyze and respond with the correct OPS")
                messages = [[SystemMessage(content=summary_prompt.to_prompt_string()), 
                    HumanMessage(content=user),
                    AIMessage(content=ai),
                    HumanMessage(content=response)]]
                patch_commands = await self._get_json_patch_commands(messages, llm)
                response = await self._preferences_resolver.apply_patch(user_id, doc, patch_commands)
                if response != "success" and self._verbose:
                    logging.warn(f"PreferencesUpdater: preferences_resolver patch application failed: {response}")