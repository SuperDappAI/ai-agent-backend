
import logging
import os
import traceback

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from jsonpatch import JsonPatch, JsonPatchException
from pydantic import BaseModel
from rate_limiter import RateLimiter
from asyncio import Lock


class QueryPreferencesInput(BaseModel):
    user_id: str


class PreferencesResolver:
    def __init__(self):
        load_dotenv()  # Load environment variables
        self.uri = os.getenv("MONGODB_URL")
        self.client = None
        self.pref_collection = None
        self.role_collection = None
        self.rate_limiter = None
        self.schema = {
            'name_nickname': "",
            'traits': [],
            'achievements': [],
            'mood_feelings': [],
            'goals': [],
            'tasks': [],
            'subtasks': [],
            'active_task_id': '',
            'active_subtask_id': '',
            'facts_opinions': [],
            'interests': [],
            'links': [],
            'skills': [],
            'occupations': [],
            'communication': {
                'data_sharing': {
                    'preferences': True,
                    'history': False
                },
                'engagement': {
                    'contact_methods': ['text', 'voice', 'video'],
                    'DND': {
                        'enabled': False,
                        'times': '22:00-06:00'
                    }
                }
            }
        }
        self.default_preferences = None
        self.init_lock = Lock()

    async def initialize(self):
        async with self.init_lock:
            if self.client is not None:
                return
            try:
                self.client = AsyncIOMotorClient(
                    self.uri, server_api=ServerApi('1'))
                await self.client.admin.command('ping')
                print("Pinged your deployment. You successfully connected to MongoDB!")

                # Setup references after successful connection
                self.db = self.client['PreferencesDB']
                self.pref_collection = self.db['Preferences']
                self.role_collection = self.db['Roles']
                self.rate_limiter = RateLimiter(rate=10, period=1)
                self.default_preferences = self.schema
            except Exception as e:
                logging.warning(
                    f"PreferencesResolver: initialize exception {e}\n{traceback.format_exc()}")

    async def get_preferences(self, user_id):
        if self.client is None or self.pref_collection is None or self.rate_limiter is None:
            await self.initialize()
        try:
            doc = await self.rate_limiter.execute(self.pref_collection.find_one, {"_id": user_id})
            if doc is None:
                await self.create_default_preferences(user_id)
                return self.default_preferences
            return doc
        except Exception as e:
            logging.warning(
                f"PreferencesResolver: get_preferences exception {e}\n{traceback.format_exc()}")
            return None

    async def get_role(self, conversation_id):
        if self.client is None or self.role_collection is None or self.rate_limiter is None:
            await self.initialize()
        try:
            roleObj = await self.rate_limiter.execute(self.role_collection.find_one, {"_id": conversation_id})
            if roleObj is not None:
                return roleObj["role"]
            else:
                return None
        except Exception as e:
            logging.warning(
                f"PreferencesResolver: get_role exception {e}\n{traceback.format_exc()}")
            return None

    async def set_role(self, role, conversation_id):
        if self.client is None or self.role_collection is None or self.rate_limiter is None:
            await self.initialize()
        try:
            roleObj = {"_id": conversation_id, "role": role}
            update_result = await self.rate_limiter.execute(self.role_collection.update_one, {"_id": conversation_id}, {"$set": roleObj}, upsert=True)
            if update_result.matched_count == 0 and update_result.upserted_id is None:
                logging.warning("No documents were inserted or updated.")
        except Exception as e:
            logging.warning(
                f"PreferencesResolver: set_role exception {e}\n{traceback.format_exc()}")
            return "failure"
        return "success"

    def get_schema(self):
        return self.schema

    async def create_default_preferences(self, user_id):
        if self.client is None or self.pref_collection is None or self.rate_limiter is None:
            await self.initialize()
        try:
            await self.rate_limiter.execute(self.pref_collection.insert_one, {
                '_id': user_id,
                **self.default_preferences
            })
        except Exception as e:
            logging.warning(
                f"PreferencesResolver: create_default_preferences exception {e}\n{traceback.format_exc()}")

    def check_for_nested_duplicates(self, value, target):
        if isinstance(target, list):
            res = value in target
            return res
        elif isinstance(target, dict):
            return any(self.check_for_nested_duplicates(value, sub_value) for sub_value in target.values())
        else:
            return False

    async def apply_patch(self, user_id, doc, patch_data):
        if self.client is None or self.pref_collection is None or self.rate_limiter is None:
            await self.initialize()
        # Make sure keys exist before applying patch
        for patch in patch_data:
            if patch["op"] in ["add", "replace"]:
                keys = patch["path"].lstrip('/').split('/')
                temp_doc = doc
                for i, key in enumerate(keys[:-1]):
                    if isinstance(temp_doc, list):
                        if key == "-":
                            if patch["op"] != "add":
                                return f"Error: '-' can only be used with 'add' operation. Patch {patch}"
                        elif not (0 <= int(key) < len(temp_doc)):
                            return f"Error: Key '{key}' does not exist in the document. Patch {patch}"
                        temp_doc = temp_doc[int(key)]
                    elif isinstance(temp_doc, dict):
                        if key not in temp_doc:
                            next_key = keys[i + 1]
                            if patch["op"] == "add":
                                if next_key == "-":
                                    # Create a new list if one does not exist
                                    temp_doc[key] = []
                                else:
                                    return f"Error: Cannot add a non-existing key without appending. Patch {patch}"
                            else:  # for 'replace' and other ops
                                return f"Error: Key '{key}' does not exist in the document. Patch {patch}"
                        temp_doc = temp_doc[key]

                    # Check for nested duplicates
                    if patch["op"] == "add":
                        if self.check_for_nested_duplicates(patch["value"], temp_doc):
                            logging.warning(
                                f"Duplicate patch {patch}, skipping...")
                            continue
                    # Check type and validity
                    if isinstance(temp_doc, list) and not keys[i + 1].isdigit() and keys[i + 1] != '-':
                        return f'Error: List indices must be integers or slices, not str. Patch {patch}'
                    elif isinstance(temp_doc, dict) and keys[i + 1].isdigit():
                        return f'Error: Dictionary keys must be strings, not integers. Patch {patch}'

                # Check the final nested key
                last_key = keys[-1]
                if isinstance(temp_doc, list) and last_key.isdigit():
                    if int(last_key) >= len(temp_doc):
                        return f"Error: Key '{last_key}' does not exist in the document. Patch {patch}"
                elif isinstance(temp_doc, list) and last_key != '-':
                    return f'Error: List indices must be integers or "-", not str. Patch {patch}'
                elif isinstance(temp_doc, dict) and last_key.isdigit():
                    return f'Error: Dictionary keys must be strings, not integers. Patch {patch}'

        # Apply the patch
        try:
            patch = JsonPatch(patch_data)
            modified_doc = patch.apply(doc)
        except JsonPatchException as e:
            return f"fail: {e}"
        except Exception as e:
            return f"An unknown exception occurred: {e}"
        try:
            # Update the database
            update_result = await self.rate_limiter.execute(self.pref_collection.update_one, {"_id": user_id}, {"$set": modified_doc})
            if update_result.modified_count == 0:
                logging.warning("No documents were updated.")
        except Exception as e:
            logging.warning(
                f"PreferencesResolver: update_one exception {e}\n{traceback.format_exc()}")
        return "success"
