import os

import pytest
from dotenv import load_dotenv
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import ScoredPoint

from functions_manager import ActionItem, FunctionInput, FunctionsManager
from rate_limiter import RateLimiter, SyncRateLimiter

rate_limiter = RateLimiter(rate=5, period=1)
rate_limiter_sync = SyncRateLimiter(rate=5, period=1)


@pytest.fixture(scope="session", autouse=True)
def create_collections():
    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
    )
    collections = ["functions"]

    for collection in collections:
        try:
            client.create_collection(
                collection_name=collection,
                vectors_config=rest.VectorParams(
                    size=1536,
                    distance=rest.Distance.COSINE,
                ),
            )
        except Exception as e:
            print(f"Collection {collection} already exists or failed to create: {e}")


class TestFunctionsManager:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        load_dotenv()
        self.functions_manager = FunctionsManager(rate_limiter, rate_limiter_sync)
        yield

    def test_transform(self):
        data = [{"name": "test", "description": "test description"}]
        category = "test_category"
        user_id = "2"

        transformed = self.functions_manager.transform(user_id, data, category)

        assert isinstance(transformed, list)
        assert all(isinstance(doc, Document) for doc in transformed)

    def test_count_tokens(self):
        functions = {
            "information_retrieval": [
                {"name": "test function", "description": "test description"}
            ],
            "communication": [
                {"name": "test function", "description": "test description"}
            ],
            "data_processing": [
                {"name": "test function", "description": "test description"}
            ],
            "sensory_perception": [
                {"name": "test function", "description": "test description"}
            ],
        }

        tokens = self.functions_manager.count_tokens(functions)

        assert isinstance(tokens, list)
        assert all(isinstance(token_dict, dict) for token_dict in tokens)

    def test_extract_name_and_category(self):
        documents = [
            ScoredPoint(
                id=1,
                payload={"name": "test1", "category": "cat1"},
                vector=[0.1, 0.2, 0.3],
                score=0.9,
                version=1,
            ),
            ScoredPoint(
                id=2,
                payload={"name": "test2", "category": "cat2"},
                vector=[0.4, 0.5, 0.6],
                score=0.8,
                version=1,
            ),
        ]

        extracted = self.functions_manager.extract_name_and_category(documents)

        expected_output = [
            {"name": "test1", "category": "cat1"},
            {"name": "test2", "category": "cat2"},
        ]

        assert extracted == expected_output

    @pytest.mark.asyncio
    async def test_pull_functions(self):
        function_input = FunctionInput(
            api_key=os.getenv("OPENAI_API_KEY"),
            action_items=[ActionItem(action="act", intent="int", category="cat")],
        )

        try:
            response, time_taken = await self.functions_manager.pull_functions(
                function_input
            )
        except UnexpectedResponse as e:
            pytest.fail(f"UnexpectedResponse: {e}")

        assert isinstance(response, list)
        assert isinstance(time_taken, float)

    @pytest.mark.asyncio
    async def test_load(self):
        response = self.functions_manager.load(api_key=os.getenv("OPENAI_API_KEY"))

        assert response is not None
        print(response)

    def test_prune_functions(self):
        try:
            result = self.functions_manager.prune_functions()
        except UnexpectedResponse as e:
            pytest.fail(f"UnexpectedResponse: {e}")

        assert result is True
