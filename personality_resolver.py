
import logging
import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from jsonpatch import JsonPatch, JsonPatchException

class PersonalityResolver:
    def __init__(self):
        load_dotenv()  # Load environment variables
        mongopw = os.getenv("MONGODB_PW")
        uri = f"mongodb+srv://superdapp:{mongopw}@cluster0.qyi8mou.mongodb.net/?retryWrites=true&w=majority"
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
        self.db = self.client['PersonalityDB']
        self.collection = self.db['Personality']

    def get_personality(self, user_id):
        doc = self.collection.find_one({"_id": user_id})
        if doc is None:
            return self.create_default_personality(user_id)
        # Filter out empty fields or fields with empty lists
        filtered_doc = {k: v for k, v in doc.items() if v not in (None, [], '')}

        return filtered_doc

    def get_schema(self):
        schema = {
            'name_nicknames': [],
            'traits': [],
            'achievements': [],
            'mood_feelings': [],
            'goals': [],
            'tasks': [
                {'id': 'task_0', 'description': 'onboard user to superdapp'},
            ],
            'subtasks': [
                {'id': 'subtask_0', 'task_id': 'task_0', 'description': 'find out user preferences, google calendar or calendly link'},
            ],
            'active_task_id': 'task_0',
            'active_subtask_id': 'subtask_0',
            'facts_opinions': [],
            'interests': [],
            'skills': [],
            'occupations': [],
            'privacy': {
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
        return schema

    def create_default_personality(self, user_id):
        # Default personality schema
        default_personality = {
            'name_nicknames': ["user name"],
            'traits': ["curious"],
            'achievements': ["became a superdapp user"],
            'mood_feelings': ["happy"],
            'goals': ["onboard to superdapp"],
            'tasks': [
                {'id': 'task_0', 'description': 'onboard user to superdapp'},
                {'id': 'task_1', 'description': 'set up tutorials'},
            ],
            'subtasks': [
                {'id': 'subtask_0', 'task_id': 'task_0', 'description': 'find out user preferences, google calendar or calendly link'},
                {'id': 'subtask_1', 'task_id': 'task_0', 'description': 'Setup web3 wallet'},
                {'id': 'subtask_2', 'task_id': 'task_0', 'description': 'See if user wants to pay SUPR to use code interpreter or social groups'},
                {'id': 'subtask_3', 'task_id': 'task_1', 'description': 'Reference docs in regards to tutorials on superdapp'},
            ],
            'active_task_id': 'task_0',
            'active_subtask_id': 'subtask_0',
            'facts_opinions': ["superdapp is awesome!"],
            'interests': ["AI", "machine learning"],
            'skills': ["hockey", "building legos"],
            'occupations': ["engineer"],
            'privacy': {
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
        
        # Insert the default personality into the collection
        self.collection.insert_one({
            '_id': user_id,
            **default_personality
        })
        
        return default_personality

    def check_for_nested_duplicates(self, value, target):
        if isinstance(target, list):
            res = value in target
            return res
        elif isinstance(target, dict):
            return any(self.check_for_nested_duplicates(value, sub_value) for sub_value in target.values())
        else:
            return False


    def apply_patch(self, user_id, doc, patch_data):
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
                                    temp_doc[key] = []  # Create a new list if one does not exist
                                else:
                                    return f"Error: Cannot add a non-existing key without appending. Patch {patch}"
                            else:  # for 'replace' and other ops
                                return f"Error: Key '{key}' does not exist in the document. Patch {patch}"
                        temp_doc = temp_doc[key]

                    # Check for nested duplicates
                    if patch["op"] == "add":
                        if self.check_for_nested_duplicates(patch["value"], temp_doc):
                            logging.warn(f"Duplicate patch {patch}, skipping...")
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

        # Update the database
        update_result = self.collection.update_one({"_id": user_id}, {"$set": modified_doc})
        if update_result.modified_count == 0:
            logging.warn("No documents were updated.")
        return "success"