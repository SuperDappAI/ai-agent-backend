import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain.schema import Document
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI
from document_summarizer import FlexibleDocumentSummarizer, SummaryPrompt


@pytest.mark.asyncio
async def test_flexible_document_summarizer():
    mock_llm = AsyncMock(ChatOpenAI)
    mock_llm.agenerate.return_value = MagicMock(generations=[[MagicMock(
        text="user summary text")], [MagicMock(text="aida summary text")]])

    summarizer = FlexibleDocumentSummarizer(llm=mock_llm, verbose=True)

    mock_document = MagicMock(Document)
    mock_document.metadata = {"summarizations": 2, "importance": "high"}
    mock_document.page_content = "{\"user\": \"user original text\", \"AiDA\": \"aida original text\"}"

    await summarizer._get_single_summary(mock_document)

    # assertions
    mock_llm.agenerate.assert_called_once()
    expected_prompt = SummaryPrompt(
        summarizations=2, importance="high").to_prompt_string()
    expected_messages = [[SystemMessage(content=expected_prompt), HumanMessage(content="user original text")], [
        SystemMessage(content=expected_prompt), HumanMessage(content="aida original text")]]
    assert mock_llm.agenerate.call_args[0][0] == expected_messages

    # Verify content was summarized
    assert mock_document.page_content == "{\"user\": \"user summary text\", \"AiDA\": \"aida summary text\"}"

    # test for multiple documents with asummarize method
    documents = [mock_document] * 5  # replace the number with your need
    await summarizer.asummarize(documents)

    # assertions
    # the method should have been called for each document
    assert mock_llm.agenerate.call_count == 6
