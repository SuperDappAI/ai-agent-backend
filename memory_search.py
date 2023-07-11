from dotenv import load_dotenv
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.text_splitter import CharacterTextSplitter  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor, DocumentCompressorPipeline
from langchain.chains import LLMChain
from langchain.schema import Document
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
            "chat-message-history", embedding=self.embeddings, namespace=self.user_id
        )
        self.memory = VectorStoreRetrieverMemory(
            retriever=self.pinecone_db.as_retriever(), memory_key="chat_history"
        )

    def push_memory(self, message, llm_response):
        start_time = time.time()
        
        self.memory.save_context({"user": message}, {"assistant": llm_response})

        time_count = time.time() - start_time
        return f"success, save_context call took {time_count:.4f} seconds"

    def split_and_push_webpage(self, html_docs):
        start_time = time.time()
        splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4096,chunk_overlap=0,disallowed_special="all")
        html_docs_split = splitter.split_documents(html_docs)
        self.pinecone_db.as_retriever().add_documents(html_docs_split)
        time_count = time.time() - start_time
        return f"success, save_context call took {time_count:.4f} seconds"

    def custom_splitter(self,docs):
        split_docs = []
        for doc in docs:
            texts = re.split("user: |\nassistant: ", doc.page_content)[1:]
            split_docs.append([Document(page_content=texts[0],metadata={'role':'user'})])
            split_docs.append([Document(page_content=texts[1],metadata={'role':'assistant'})])
        return [split_docs]


    def get_relevant_memory_docs(self, query, context, deepSearch=False):
        start_time = time.time()
        retriever = self.pinecone_db.as_retriever(search_kwargs={"k": 10})
        token_text_splitter= CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256, chunk_overlap=0
        ) 
        embeddings_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=0.72,k=self.k_num) 
        llm = ChatOpenAI(temperature=0,verbose=True)

        CONTEXT_PROMPT = PromptTemplate(
        input_variables=["question","context"],
        
        template="""You are an AI language model assistant.
        Your task is to generate a question to ask back to the user for clarification of a missing piece of context. The question should be concise but clear.
        Missing context: {context}
        Original question: {question}""",
        )

        memories = []

        if not deepSearch:

            memory_docs = retriever.get_relevant_documents(query)
            docs_with_metadata = []
            for doc in memory_docs:
                texts = re.split("user: |\nassistant: ", doc.page_content)[1:]
                docs_with_metadata.append(Document.construct(page_content=texts[0],metadata={'role':'user'}))
                docs_with_metadata.append(Document.construct(page_content=texts[1],metadata={'role':'assistant'}))

            docs_split = token_text_splitter.split_documents(docs_with_metadata)
            
            docs_filtered = embeddings_filter.compress_documents(docs_split, context)
            if docs_filtered[0].page_content == "":
                docs_filtered = docs_split
            docs_formatted = []
            for doc in docs_filtered:
                docs_formatted.append({"content": doc.page_content, "metadata": doc.metadata})
            memories.append({"context": docs_formatted})

        else: 
            memory_docs = retriever.get_relevant_documents(query)
            docs_with_metadata = []
            for doc in memory_docs:
                texts = re.split("user: |\nassistant: ", doc.page_content)[1:]
                docs_with_metadata.append(Document.construct(page_content=texts[0],metadata={'role':'user'}))
                docs_with_metadata.append(Document.construct(page_content=texts[1],metadata={'role':'assistant'}))

            docs_split = token_text_splitter.split_documents(docs_with_metadata)
            
            docs_filtered = embeddings_filter.compress_documents(docs_split, context)
            if docs_filtered[0].page_content == "":
                docs_filtered = docs_split

            llm = ChatOpenAI(temperature=0)
            extractor = LLMChainExtractor.from_llm(llm)
            context_chain=LLMChain(llm=OpenAI(temperature=0,verbose=True),prompt=CONTEXT_PROMPT)
            context_query = context_chain({"question": query, "context": context})
            print(context_query)
            docs_extracted = extractor.compress_documents(docs_filtered, context_query['text'])

            docs_formatted = []
            for doc in docs_extracted:
                docs_formatted.append({"content": doc.page_content, "metadata": doc.metadata})
            memories.append({"context": docs_formatted})

        time_count = time.time() - start_time
        return memories, f"success, retrieve call took {time_count:.4f} seconds"

    def semantic_search_html(self, query, context, similarity_threshold): 
        start_time = time.time()
        retriever = self.pinecone_db.as_retriever(search_kwargs={"k": 10})
        token_text_splitter= CharacterTextSplitter(
        chunk_size=512, chunk_overlap=0
        ) 
        embeddings_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=similarity_threshold,k=self.k_num) 

        memories = []

        memory_docs = retriever.get_relevant_documents(query)

        docs_split = token_text_splitter.split_documents(memory_docs)
        
        docs_filtered = embeddings_filter.compress_documents(docs_split, query=f'{query},{context}')
        if len(docs_filtered) == 0:
            try:
                docs_filtered = docs_split[0:3]
            except:
                return [], "No results found"
        docs_formatted = []
        for doc in docs_filtered:
            docs_formatted.append({"content": doc.page_content, "metadata": doc.metadata})
        memories.append({"context": docs_formatted})

        time_count = time.time() - start_time
        return memories, f"success, retrieve call took {time_count:.4f} seconds"

    def get_functions(self, action, num_results):
        start_time = time.time()
        # retriever = self.pinecone_db.as_retriever(search_kwargs={"k": num_results, "metadata": {"category": category}})
        retriever = self.pinecone_db.as_retriever(search_kwargs={"k": num_results})
        func_docs = retriever.get_relevant_documents(action)
        result = []
        for doc in func_docs:
            result.append({"name": doc.metadata["name"], "category": doc.metadata["category"]})
        time_count = time.time() - start_time
        return result, f"success, retrieve call took {time_count:.4f} seconds"

    def get_user_id(self):
        return self.user_id

    def set_user_id(self, user_id):
        self.user_id = user_id

    def clear_user_memory(self):
        native_index_object = pinecone.Index("chat-message-history")
        native_index_object.delete(namespace=self.user_id, delete_all=True)
        return f"Memories cleared for user: {self.user_id}"
