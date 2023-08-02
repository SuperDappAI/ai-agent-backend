from dotenv import load_dotenv
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
import time
import re

load_dotenv()

class MemoryManager():
    def __init__(self, user_id):
        self.user_id = user_id
        self.embeddings = OpenAIEmbeddings()
        self.pinecone_db = Pinecone.from_existing_index("chat-message-history",embedding=self.embeddings,namespace=self.user_id)
        self.memory = VectorStoreRetrieverMemory(retriever=self.pinecone_db.as_retriever(), memory_key="chat_history")


    def push_memory(self, message, llm_response):
        start_time = time.time()
        self.memory.save_context({"user": message}, {"assistant": llm_response})
        
        time_count = time.time() - start_time
        return f"success, save_context call took {time_count:.4f} seconds"

    def get_relevant_memory_docs(self,query):
        start_time = time.time()
        retriever = self.pinecone_db.as_retriever()
        memory_docs = retriever.get_relevant_documents(query)
        memories = [] 
        for doc in memory_docs:
            my_string = doc.page_content
            split_string = re.split('user: |\nassistant: ', my_string)[1:]
            memories.append({'user': split_string[0], 'assistant': split_string[1]})
            
        time_count = time.time() - start_time
        return memories, f"success, retrieve call took {time_count:.4f} seconds"

    def get_user_id(self):
        return self.user_id

    def set_user_id(self, user_id):
        self.user_id = user_id

    def clear_user_memory(self):
        native_index_object = pinecone.Index("chat-message-history")
        native_index_object.delete(namespace=self.user_id,delete_all=True)
        return f"Memories cleared for user: {self.user_id}"
