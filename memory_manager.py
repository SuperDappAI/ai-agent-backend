from dotenv import load_dotenv
import time
import shutil
import schedule
from llama_index import OpenAI, ServiceContext, LLMRerank, Document, VectorStoreIndex, StorageContext, load_index_from_storage

load_dotenv()

class MemoryManager1:
    def __init__(self):
        self.dirpath = "./memory"
        self.index = []
        self.query_engine = []
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, service_context=ServiceContext.from_defaults(
            llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
        ))
        schedule.every(30).to(60).seconds.do(self.save)

    def load(self, user_id):
        print("MemoryManager: Loading from disk")
        start = time.time()
        userpath = f"{self.dirpath}_{user_id}"
        if userpath.exists() and userpath.is_dir():
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=userpath)

            # load index
            self.index[user_id] = load_index_from_storage(storage_context)
            self.query_engine[user_id] = self.index[user_id].as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[self.reranker],
                response_mode="tree_summarize"
            )
        end = time.time()
        print(f"MemoryManager: Load operation took {end - start} seconds")

    def save(self):
        start = time.time()
        self.saving = True
        for idx, doc in enumerate(self.index):
            if doc.dirty is True:
                filepath = f"{self.dirpath}_{idx}"
                doc.storage_context.persist(persist_dir=filepath)
                doc.dirty = False
        self.saving = False
        end = time.time()
        print(f"MemoryManager: Save operation took {end - start} seconds")

    def push_memory(self, user_id, query, llm_response):
        start = time.time()
        if user_id not in self.index:
            self.load(user_id)
        doc = Document({"user": query}, {"assistant": llm_response})
        # if not loaded because it didn't exist on disk, then create a new one otherwise just upsert new doc
        if user_id not in self.index:
            self.index[user_id] = VectorStoreIndex.from_documents([doc])
        else:
            self.index[user_id].update(doc)
        self.index[user_id].dirty = True
        self.query_engine[user_id] = self.index[user_id].as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[self.reranker],
            response_mode="tree_summarize"
        )
        end = time.time()
        print(f"MemoryManager: push_memory operation took {end - start} seconds")

    def pull_memory(self, user_id, query):
        start = time.time()
        response = None
        if user_id not in self.query_engine:
            response = self.query_engine[user_id].query(
                query, 
            )
        end = time.time()
        print(f"MemoryManager: pull_memory operation took {end - start} seconds")
        return response

    def delete_memory(self, user_id):
        userpath = f"{self.dirpath}_{user_id}"
        if userpath.exists() and userpath.is_dir():
            shutil.rmtree(userpath)
        self.index[user_id] = None