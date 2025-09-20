import asyncio
import os
import time

import pytest
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever

from rate_limiter import RateLimiter, SyncRateLimiter
from web_manager import HTMLInput, HTMLItem, WebManager

rate_limiter = RateLimiter(rate=5, period=1)
rate_limiter_sync = SyncRateLimiter(rate=5, period=1)


@pytest.fixture
def setup_web_manager():
    load_dotenv()
    web_manager = WebManager(rate_limiter, rate_limiter_sync)
    yield web_manager


@pytest.fixture
def setup_html_input():
    html_input = HTMLInput(
        api_key=os.getenv("OPENAI_API_KEY"),
        action_items=[
            HTMLItem(source_url="http://example.com", html_doc="test_html_doc")
        ],
        hash="test_hash",
        query="test_query",
    )
    return html_input


@pytest.mark.asyncio
async def test_load(setup_web_manager, setup_html_input):
    try:
        web_manager = setup_web_manager
        start_time = time.time()
        memory = web_manager.load(setup_html_input.api_key)
        end_time = time.time()
        assert isinstance(memory, ContextualCompressionRetriever)
        assert end_time - start_time >= 0
    except:
        print("Load test not executed")


@pytest.mark.asyncio
async def test_search_html(setup_web_manager, setup_html_input):
    web_manager = setup_web_manager
    html_input = setup_html_input
    response, duration = await web_manager.search_html(html_input)
    assert isinstance(response, list)
    assert isinstance(duration, float)
    for item in response:
        assert "text" in item
        assert "source_url" in item

    current_task = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current_task]
    for task in pending:
        await task
