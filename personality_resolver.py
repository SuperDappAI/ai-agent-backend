import jsonpatch
import time
import logging
import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

class JsonPatchOperation(BaseModel):
    op: str
    path: str
    value: Optional[str] = None

class JsonPatchData(BaseModel):
    user_id: str
    json_patch_data: List[JsonPatchOperation]

class QueryFieldsInput(BaseModel):
    user_id: str
    paths: List[str] = Field(..., description="List of JSON Pointer paths to fetch")

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

    def json_pointer_to_dot_notation(self, json_pointer):
        # Remove the leading slash and replace remaining slashes with dots
        return json_pointer.lstrip('/').replace('/', '.')

    def get_fields(self, user_id, fields):
        start = time.time()
        # Convert JSON Pointer paths to MongoDB dot-notation paths
        dot_notation_fields = {self.json_pointer_to_dot_notation(field): 1 for field in fields}
        doc = self.collection.find_one({"_id": user_id}, projection=dot_notation_fields)
        end = time.time()
        return doc, end - start

    def apply_patch(self, user_id, patch_data):
        start = time.time()
        doc = self.collection.find_one({"_id": user_id})
        if doc is None:
            empty_user = {
                "_id": user_id,
                "Aida": {},
                "User": {}
            }
            self.collection.insert_one(empty_user)
            doc = empty_user
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
            logging.Warn(f"JSON Patch failed: {e}")
            end = time.time()
            return "fail", end - start

        update_result = self.collection.update_one({"_id": user_id}, {"$set": modified_doc})
        if update_result.modified_count == 0:
            logging.Warn("No documents were updated.")
        end = time.time()
        logging.info(
            f"PersonalityResolver: apply_patch operation took {end - start} seconds")
        return "success", end - start