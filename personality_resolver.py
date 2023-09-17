
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
        return doc

    def create_default_personality(self, user_id):
        # Default personality schema
        default_personality = {
            'name_nicknames': ["user name"],
            'traits': ["curious"],
            'achievements': ["became a superdapp user"],
            'mood_feelings': ["happy"],
            'goals': ["onboard to superdapp"],
            'tasks': [
                {
                    'task': 'onboard user to superdapp',
                    'active': True,
                    'subtasks': [
                        {
                            'subtask': 'find out user preferences, google calendar or calendly link',
                            'active': True
                        },
                        {
                            'subtask': 'Setup web3 wallet',
                            'active': False
                        },
                        {
                            'subtask': 'See if user wants to pay SUPR to use code interpreter or social groups',
                            'active': False
                        }
                    ]
                }
            ],
            'facts_opinions': ["superdapp is awesome!"],
            'interests': ["AI", "machine learning"],
            'occupations': ["engineer"],
            'privacy': {
                'data_sharing': {
                    'anonymous': True,
                    'personal': False,
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

    def apply_patch(self, user_id, doc, patch_data):
        # Make sure keys exist before applying patch
        for patch in patch_data:
            if patch["op"] in ["add", "replace"]:
                keys = patch["path"].lstrip('/').split('/')
                temp_doc = doc
                for i, key in enumerate(keys[:-1]):
                    if key not in temp_doc:
                        next_key = keys[i + 1]
                        if next_key == "-":
                            return "Error: Attempting to append to a non-existent list."
                        else:
                            return f"Error: Key '{key}' does not exist in the document."
                    temp_doc = temp_doc[key]

                    # Check type and validity
                    if isinstance(temp_doc, list) and not keys[i + 1].isdigit() and keys[i + 1] != '-':
                        return 'Error: List indices must be integers or slices, not str'
                    elif isinstance(temp_doc, dict) and keys[i + 1].isdigit():
                        return 'Error: Dictionary keys must be strings, not integers'

                # Check the final nested key
                last_key = keys[-1]
                if isinstance(temp_doc, list) and not last_key.isdigit() and last_key != '-':
                    return 'Error: List indices must be integers or "-", not str'
                elif isinstance(temp_doc, dict) and last_key.isdigit():
                    return 'Error: Dictionary keys must be strings, not integers'

        # Apply the patch
        try:
            patch = JsonPatch(patch_data)
            modified_doc = patch.apply(doc)
        except JsonPatchException as e:
            return f"fail: {e}"

        # Update the database
        update_result = self.collection.update_one({"_id": user_id}, {"$set": modified_doc})
        if update_result.modified_count == 0:
            logging.warn("No documents were updated.")
        return "success"