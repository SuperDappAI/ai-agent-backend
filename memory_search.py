from dotenv import load_dotenv
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor, DocumentCompressorPipeline
from langchain.chains import LLMChain
from langchain.schema import Document
import asyncio
import time
import re
from langchain.prompts import PromptTemplate

load_dotenv()


class MemoryManager:
    def __init__(self, user_id, k_num=5):
        self.user_id = user_id
        self.k_num = k_num
        self.embeddings = OpenAIEmbeddings()
        self.pinecone_db = Pinecone.from_existing_index(
            "aida", embedding=self.embeddings, namespace=self.user_id
        )
        self.memory = VectorStoreRetrieverMemory(
            retriever=self.pinecone_db.as_retriever(), memory_key="chat_history"
        )

    def push_memory(self, message, llm_response):
        start_time = time.time()

        self.memory.save_context(
            {"user": message}, {"assistant": llm_response})

        time_count = time.time() - start_time
        return f"success, save_context call took {time_count:.4f} seconds"

    def split_and_push_webpage(self, docs):
        start_time = time.time()
        # splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4096,chunk_overlap=0,disallowed_special="all")
        splitter = TokenTextSplitter(chunk_size=4096, chunk_overlap=0)
        print(f'time for split {time.time() - start_time}')
        docs_split = splitter.split_documents(docs)
        # self.pinecone_db.as_retriever().add_documents(docs_split)
        self.pinecone_db.add_documents(docs_split)
        print(f'time for push {time.time() - start_time}')
        time_count = time.time() - start_time
        return f"success, save_context call took {time_count:.4f} seconds"

    def custom_splitter(self, docs):
        split_docs = []
        for doc in docs:
            texts = re.split("user: |\nassistant: ", doc.page_content)[1:]
            split_docs.append(
                [Document(page_content=texts[0], metadata={'role': 'user'})])
            split_docs.append(
                [Document(page_content=texts[1], metadata={'role': 'assistant'})])
        return [split_docs]

    def get_relevant_memory_docs(self, query, context, num_chunks, num_neighbors, similarity_threshold):
        start_time = time.time()
        retriever = self.pinecone_db.as_retriever(search_kwargs={"k": 15})
        token_text_splitter = TokenTextSplitter(
            chunk_size=256, chunk_overlap=0
        )
        embeddings_filter = EmbeddingsFilter(
            embeddings=OpenAIEmbeddings(), similarity_threshold=similarity_threshold, k=num_chunks)
        memories = []

        memory_docs = retriever.get_relevant_documents(query)
        docs_with_metadata = []
        index = 0
        for doc in memory_docs:
            texts = re.split("user: |\nassistant: ", doc.page_content)[1:]
            docs_with_metadata.append(Document.construct(
                page_content=texts[0], metadata={'role': 'user', 'index': index}))
            docs_with_metadata.append(Document.construct(
                page_content=texts[1], metadata={'role': 'assistant','index': index}))
            index += 1

        docs_split = token_text_splitter.split_documents(
            docs_with_metadata)

        inner_index = 0
        for doc in docs_split:
            doc.metadata['inner_index'] = inner_index 
            inner_index += 1

        docs_filtered = embeddings_filter.compress_documents(
            docs_split, context)
        
        if docs_filtered[0].page_content == "":
            docs_filtered = docs_split[0:num_chunks]
            docs_formatted = []
        else:
            docs_formatted = []
            for i, doc in enumerate(docs_filtered):
                matching_docs = [d for d in docs_split if (d.metadata['index'] == doc.metadata['index'])]
                if i - num_neighbors < 0:
                    previous_docs = matching_docs[0:i-1]
                elif i - num_neighbors >= 0:
                    previous_docs = matching_docs[i - num_neighbors:i-1]
                if i + num_neighbors >= len(matching_docs): 
                    subsequent_docs = matching_docs[i+1:]
                elif i + num_neighbors < len(matching_docs):
                    subsequent_docs = matching_docs[i+1:i+num_neighbors+1]
                docs_formatted.append(
                    {"Matched Content": doc.page_content, "Matched Role": doc.metadata['role'], "Previous Neighbors": previous_docs, "Subsequent Neighbors": subsequent_docs})
                memories.append({"context": docs_formatted})

        time_count = time.time() - start_time
        return memories, f"success, retrieve call took {time_count:.4f} seconds"

    def semantic_search_html(self, query, context, similarity_threshold):
        start_time = time.time()
        retriever = self.pinecone_db.as_retriever(search_kwargs={"k": 10})
        token_text_splitter = TokenTextSplitter(
            chunk_size=256, chunk_overlap=0
        )
        embeddings_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(
        ), similarity_threshold=similarity_threshold, k=self.k_num)

        memories = []

        memory_docs = retriever.get_relevant_documents(query)

        docs_split = token_text_splitter.split_documents(memory_docs)

        docs_filtered = embeddings_filter.compress_documents(
            docs_split, query=f'{query},{context}')
        if len(docs_filtered) == 0:
            try:
                docs_filtered = docs_split[0:3]
            except:
                return [], "No results found"
        docs_formatted = []
        for doc in docs_filtered:
            docs_formatted.append(
                {"content": doc.page_content, "metadata": doc.metadata})
        memories.append({"context": docs_formatted})

        time_count = time.time() - start_time
        return memories, f"success, retrieve call took {time_count:.4f} seconds"

    async def get_functions(self, actions, intent, categories, num_results, similarity_threshold):
        start_time = time.time()


        retriever = self.pinecone_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={
                                                "k": num_results, "score_threshold": similarity_threshold})

        async def get_docs(action):
            func_docs = await retriever.aget_relevant_documents(f'{action}.{intent}.{categories}')
            if not func_docs:
                return [
                    # leave this commented out for now, might add another fallback later
                    # {
                    #     "name": "searchWebGeneral",
                    #     "category": "informationretrieval_functions"
                    # }
                ]
            else:
                return [
                    {"name": doc.metadata["name"], "category": doc.metadata["category"]}
                    for doc in func_docs
                ]

        tasks = [get_docs(action) for action in actions]
        results = await asyncio.gather(*tasks)
        result = [item for sublist in results for item in sublist]

        # Removing duplicates by converting list of dictionaries to dictionary and back to list
        result = [dict(t) for t in set(tuple(i.items()) for i in [item for sublist in results for item in sublist])]

        time_count = time.time() - start_time
        print(time_count)


        return result, f"success, retrieve call took {time_count:.4f} seconds"

    def get_user_id(self):
        return self.user_id

    def set_user_id(self, user_id):
        self.user_id = user_id

    def clear_user_memory(self):
        native_index_object = pinecone.Index("aida")
        native_index_object.delete(namespace=self.user_id, delete_all=True)
        return f"Memories cleared for user: {self.user_id}"
