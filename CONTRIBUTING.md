# Contributing to SuperDappAI

Thank you for your interest in contributing to SuperDappAI! We welcome contributions from the community and are pleased to have you join us.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps which reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include screenshots if helpful**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain which behavior you expected to see instead**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `dev`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- OpenAI API key (for testing)
- Qdrant instance (local or cloud)
- MongoDB instance (local or cloud)

### Setting Up Your Environment

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/AI.git
   cd AI
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run tests to ensure everything works**

   ```bash
   python -m pytest tests/ -v
   ```

## Development Guidelines

### Code Style

We follow Python best practices:

- **PEP 8**: Follow the Python style guide
- **Type hints**: Use type hints where appropriate
- **Docstrings**: Document functions and classes
- **Async/await**: Use async patterns for I/O operations

### Testing

- Write tests for new features
- Ensure tests pass before submitting PR
- Use mocking for external services when appropriate
- Aim for good test coverage

### Commit Messages

Write clear, concise commit messages:

```
type(scope): short description

Longer description if needed

- List any breaking changes
- Reference issues: Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix  
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/tooling changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-endpoint`
- `fix/memory-leak-issue`
- `docs/update-readme`
- `test/add-integration-tests`

## Project Structure

```
├── main.py                 # FastAPI application entry point
├── agent_manager.py        # Core agent functionality
├── functions_manager.py    # Function management system
├── doc_manager.py         # Document processing
├── web_manager.py         # Web content processing
├── memory_summarizer.py   # Memory optimization
├── preferences_resolver.py # User preferences
├── rate_limiter.py        # Rate limiting utilities
├── tests/                 # Test suite
├── docker/                # Docker configuration
└── requirements.txt       # Dependencies
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=.

# Run specific test file
python -m pytest tests/test_specific.py

# Run with verbose output
python -m pytest -v
```

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions  
3. **API Tests**: Test HTTP endpoints
4. **Mock Tests**: Test with external service mocks

### Writing Tests

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_async_function():
    # Test async functionality
    result = await some_async_function()
    assert result == expected_value

@patch('module.external_service')
def test_with_mock(mock_service):
    # Test with mocked external dependency
    mock_service.return_value = "mocked_response"
    result = function_using_service()
    assert result == "expected_result"
```

## Documentation

### API Documentation

- Update docstrings for new endpoints
- Ensure examples are accurate
- Update OpenAPI schema if needed

### Code Documentation

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        SpecificError: When this error occurs
    """
    pass
```

## Performance Considerations

- Use async/await for I/O operations
- Implement proper caching strategies
- Consider rate limiting impacts
- Monitor memory usage
- Profile performance-critical code

## Security Guidelines

- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all inputs
- Implement proper error handling
- Follow security best practices for APIs

## Questions?

- Check existing issues and discussions
- Ask questions in GitHub Discussions
- Reach out to maintainers

Thank you for contributing to SuperDappAI!