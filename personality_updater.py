
import traceback
import logging
import json
import re

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from personality_resolver import PersonalityResolver
from typing import List

class SystemPrompt:
    def __init__(self, personality: str, error: str = ""):
         self.personality = personality
         self.error = error

    def to_prompt_string(self) -> str:
        template_text = f"""
        You are a top-tier preferences interpreter. Your job is to take the existing PREFERENCES and assess the user / AI dialog to infer new unique, case-insensitive user preferences. Only include attributes that are important and likely to be referenced in future conversations with the user. Use the user's query for static attributes like name and occupation and both the user's query and the LLM's response for dynamic attributes like tasks and mood. 

        1. Provide a list of JSONPatch operations (OPS) for updating the PREFERENCES. Follow this format:
            - 'add': '/traits/-'
            - 'remove' or 'replace': '/traits/[index]'

        {self.error}

        Note:
        - Check values in OPS to make sure they are not already in PREFERENCES already.
        - Use the '-' symbol ONLY with 'add' to indicate appending to an array.
        - Indices are 0-based.
        - Skip updates for code-related or non-conversational exchanges
        - Feel free to make new field names or add to PREFERENCES schema.
        - Use 'replace' ONLY for changing an existing value at a specified index.
        - Tasks and subtasks are now managed through their IDs. Make sure to cross-reference them appropriately.
        - Only one active task or subtask is allowed at a time.
        - Use proper JSONPatch formatting.
        - Returning empty list is perfectly reasonable
        - Must be confident that each OP relates to the conversation meaningfully (or to reduce preferences size) to include it.

        The OPS applied will create a new preferences on the backend make sure this will not exceed 1000 tokens. Remove any redundant attributes to meet this requirement.

        Examples of OPS only to learn of formatting:
            OPS: []
            OPS: [{{"op": "add", "path": "/traits/-", "value": "adventurous"}},{{"op": "remove", "path": "/traits/1"}},{{"op": "replace", "path": "/traits/0", "value": "meticulous"}}]
            OPS: [{{"op": "add", "path": "/tasks/-", "value": {{"id": "task_0", "description": "New Task"}}}},{{"op": "add", "path": "/subtasks/-", "value": {{"id": "subtask_0", "task_id": "task_0", "description": "New Subtask"}}}},{{"op": "replace", "path": "/active_task_id", "value": "task_0"}},{{"op": "replace", "path": "/active_subtask_id", "value": "subtask_0"}}]
            OPS: [{{"op": "add", "path": "/achievements/-", "value": "New Achievement"}},{{"op": "add", "path": "/skills/-", "value": "New Skill"}},{{"op": "replace", "path": "/mood_feelings/0", "value": "content"}}]

        PREFERENCES: {self.personality}
        """
        return template_text

class PersonalityUpdater:
    _personality_resolver: PersonalityResolver
    _verbose: bool

    def __init__(self, personality_resolver: PersonalityResolver, verbose: bool = False) -> None:
        self._personality_resolver = personality_resolver
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
                logging.warn(f"PersonalityUpdater: _get_json_patch_commands exception, e: {e}\n{traceback.format_exc()}")

        return array_json

    async def update_personality(self, llm: ChatOpenAI, user: str, ai: str, user_id: str):
        """Reflect on recent observations and generate 'insights'."""
        doc = self._personality_resolver.get_personality(user_id)
        if doc is None:
            logging.warn(f"get_personality got empty doc")
            return
        summary_prompt = SystemPrompt(doc)
        messages = [[SystemMessage(content=summary_prompt.to_prompt_string()), 
                    HumanMessage(content=user),
                    AIMessage(content=ai)]]
        patch_commands = await self._get_json_patch_commands(messages, llm)
        if len(patch_commands) > 0:
            if self._verbose:
                logging.info("AiDA is trying to update personality")
            response = self._personality_resolver.apply_patch(user_id, doc, patch_commands)
            if response != "success":
                summary_prompt = SystemPrompt(doc, "2. You have been given human feedback that your changes were not accepted due to syntax, you are to carefully analyze and respond with the correct OPS")
                messages = [[SystemMessage(content=summary_prompt.to_prompt_string()), 
                    HumanMessage(content=user),
                    AIMessage(content=ai),
                    HumanMessage(content=response)]]
                patch_commands = await self._get_json_patch_commands(messages, llm)
                response = self._personality_resolver.apply_patch(user_id, doc, patch_commands)
                if response != "success" and self._verbose:
                    logging.warn(f"PersonalityUpdater: personality_resolver patch application failed: {response}")