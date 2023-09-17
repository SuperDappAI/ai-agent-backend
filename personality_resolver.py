import jsonpatch
import time
import logging
import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

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
            'name_nickname': [],
            'traits': [],
            'achievements': [],
            'mood_feelings': [],
            'goals': [],
            'tasks': [
                {
                    'task': '',
                    'active': False,
                    'subtasks': [
                        {
                            'subtask': '',
                            'active': False
                        }
                    ]
                }
            ],
            'facts_opinions': [],
            'expertise': [],
            'occupations': [],
            'privacy': {
                'data_sharing': {
                    'anonymous': True,
                    'personal': False,
                    'history': False
                },
                'engagement': {
                    'contact': ['text', 'voice'],
                    'DND': {
                        'enabled': False,
                        'times': ['22:00-06:00']
                    }
                }
            }
        }
        
        # Insert the default personality into the collection
        self.collection.insert_one({
            '_id': user_id,
            'personality': default_personality
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
                            temp_doc[key] = []
                        else:
                            temp_doc[key] = {}
                    temp_doc = temp_doc[key]

                # Create the final nested key if it doesn't exist
                last_key = keys[-1]
                if isinstance(temp_doc, list):
                    if last_key == "-":
                        # Append a None value at the end of the list
                        temp_doc.append(None)
                    else:
                        # Insert a None value at the specified index, assuming the index is an integer
                        index = int(last_key)
                        while len(temp_doc) <= index:
                            temp_doc.append(None)
                else: # Assume it's a dictionary
                    if last_key not in temp_doc and last_key != "-":
                        temp_doc[last_key] = None


        try:
            patch = jsonpatch.JsonPatch(patch_data)
            modified_doc = patch.apply(doc)
        except jsonpatch.JsonPatchException as e:
            return f"fail: {e}"

        update_result = self.collection.update_one({"_id": user_id}, {"$set": modified_doc})
        if update_result.modified_count == 0:
            logging.Warn("No documents were updated.")
        return "success"