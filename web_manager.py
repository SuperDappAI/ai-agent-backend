from dotenv import load_dotenv
import time
import shutil
import schedule
from llama_index import OpenAI, ServiceContext, LLMRerank, Document, VectorStoreIndex, StorageContext, load_index_from_storage

load_dotenv()


class WebManager:
    def __init__(self):
        self.dirpath = "./web"
        self.index = []
        self.query_engine = []
        self.reranker = LLMRerank(choice_batch_size=5, top_n=3, service_context=ServiceContext.from_defaults(
            llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
        ))
        schedule.every(300).to(600).seconds.do(self.save)

    def load(self, hash):
        print("WebManager: Loading from disk")
        start = time.time()
        userpath = f"{self.dirpath}_{hash}"
        if userpath.exists() and userpath.is_dir():
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=userpath)

            # load index
            self.index[hash] = load_index_from_storage(storage_context)
            self.query_engine[hash] = self.index[hash].as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[self.reranker],
                response_mode="tree_summarize"
            )
        end = time.time()
        print(f"WebManager: Load operation took {end - start} seconds")

    def save(self):
        start = time.time()
        self.saving = True
        for idx, doc in enumerate(self.index):
            if doc.dirty is True:
                doc.storage_context.persist(persist_dir=f"{self.dirpath}_{idx}")
                doc.dirty = False
        self.saving = False
        end = time.time()
        print(f"WebManager: Save operation took {end - start} seconds")

    def push_html(self, hash, urls, html_docs):
        if hash in self.index:
            print("WebManager: Error push_html, hash already exists")
            return
        start = time.time()
        documents = [Document(t) for t in html_docs]
        self.index[hash] = VectorStoreIndex.from_documents(documents)
        for idx, doc in enumerate(documents):
            doc.extra_info.url = urls[idx]
        self.index[hash].dirty = True
        self.query_engine[hash] = self.index[hash].as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[self.reranker],
            response_mode="tree_summarize"
        )
        end = time.time()
        print(f"WebManager: push_html operation took {end - start} seconds")

    def pull_html(self, hash, query):
        if hash not in self.query_engine:
            print("WebManager: Error pull_html, hash doesn't exists")
            return None
        start = time.time()
        response = self.query_engine[hash].query(
            query, 
        )
        end = time.time()
        print(f"WebManager: pull_html operation took {end - start} seconds")
        return response

    def delete_memory(self, hash):
        hashpath = f"{self.dirpath}_{hash}"
        if hashpath.exists() and hashpath.is_dir():
            shutil.rmtree(hashpath)
        self.index[hash] = None