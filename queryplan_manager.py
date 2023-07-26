import time
import os
from dotenv import load_dotenv
from llama_index import ServiceContext, LLMPredictor
from llama_index.llms import OpenAI
from llama_index.indices.empty.base import EmptyIndex
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.query_engine.multistep_query_engine import MultiStepQueryEngine


class QueryPlanManager:
    def __init__(self):
        load_dotenv()  # Load environment variables
        os.getenv("OPENAI_API_KEY")  # Get API Key from environment variable

        # LLMPredictor (gpt-4)
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4"))
        service_context = ServiceContext.from_defaults(llm=llm_predictor)
        # gpt-4
        step_decompose_transform = StepDecomposeQueryTransform(
            llm_predictor, verbose=True
        )
        index = EmptyIndex()
        self.query_engine = index.as_query_engine(
            service_context=service_context
        )
        self.query_engine = MultiStepQueryEngine(self.query_engine, query_transform=step_decompose_transform)

    def query_plan(self, query):
        start = time.time()
        response = self.query_engine.query(query)
        end = time.time()
        print(f"QueryPlanManager: query_plan operation took {end - start} seconds")
        return response
