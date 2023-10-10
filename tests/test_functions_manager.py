
import pytest
import os
from functions_manager import FunctionsManager, ActionItem, FunctionInput
from dotenv import load_dotenv
from langchain.schema import Document
from rate_limiter import RateLimiter
rate_limiter = RateLimiter(rate=5, period=1)

class TestFunctionsManager:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        load_dotenv()
        self.functions_manager = FunctionsManager(rate_limiter)
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
            "information_retrieval": [{"name": "test function", "description": "test description"}],
            "communication": [{"name": "test function", "description": "test description"}],
            "data_processing": [{"name": "test function", "description": "test description"}],
            "sensory_perception": [{"name": "test function", "description": "test description"}],
        }

        tokens = self.functions_manager.count_tokens(functions)

        assert isinstance(tokens, list)
        assert all(isinstance(token_dict, dict) for token_dict in tokens)

    def test_extract_name_and_category(self):
        documents = [
            Document(
                page_content='{"name": "test1", "category": "cat1"}', metadata={}),
            Document(
                page_content='{"name": "test2", "category": "cat2"}', metadata={})
        ]

        extracted = self.functions_manager.extract_name_and_category(documents)

        expected_output = [{'name': 'test1', 'category': 'cat1'}, {
            'name': 'test2', 'category': 'cat2'}]

        assert extracted == expected_output

    @pytest.mark.asyncio
    async def test_pull_functions(self):
        function_input = FunctionInput(api_key=os.getenv("OPENAI_API_KEY"),
                                       action_items=[ActionItem(
                                           action="act", intent="int", category="cat")]
                                       )

        response, time_taken = await self.functions_manager.pull_functions(function_input)

        assert isinstance(response, list)
        assert isinstance(time_taken, float)

    @pytest.mark.asyncio
    async def test_load(self):
        response = self.functions_manager.load(
            api_key=os.getenv("OPENAI_API_KEY"))

        assert response is not None
        print(response)

    def test_prune_functions(self):
        result = self.functions_manager.prune_functions()

        assert result is True
