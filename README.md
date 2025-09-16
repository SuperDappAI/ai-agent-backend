# SuperDappAI - AI Agent Platform

A comprehensive AI agent platform built with FastAPI that provides intelligent memory management, document processing, and function orchestration capabilities. This platform enables the creation of sophisticated AI agents with long-term memory, context-aware responses, and extensible functionality.

## Features

- **Intelligent Memory Management**: Long-term memory with decay functions and conversation summarization
- **Document Processing**: Semantic search and document management with vector storage
- **Function Management**: Dynamic function discovery and orchestration
- **Web Content Processing**: HTML content analysis and semantic search
- **Multi-Modal Support**: Text, document, and web content processing
- **Rate Limiting**: Built-in rate limiting for API calls and resource management
- **Caching**: Intelligent caching system for improved performance
- **User Preferences**: Personalized user preference management

## Architecture

The platform consists of several key components:

- **Agent Manager**: Core agent functionality with memory management
- **Functions Manager**: Dynamic function discovery and execution
- **Document Manager**: Document ingestion and semantic search
- **Web Manager**: Web content processing and analysis  
- **Memory Summarizer**: Conversation summarization and memory optimization
- **Preferences Resolver**: User preference management and personalization

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Qdrant vector database (cloud or self-hosted)
- MongoDB for preferences storage
- Cohere API key (optional, for reranking)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SuperDappAI/AI.git
cd AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=your_qdrant_url_here
MONGODB_URL=your_mongodb_connection_string
COHERE_API_KEY=your_cohere_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
CONSOLE_KEY=your_console_key_for_admin_operations
```

### 4. Run the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

### Key Endpoints

- **POST /push_memory/**: Store memories for a user
- **POST /pull_memory/**: Retrieve relevant memories
- **POST /push_functions/**: Register new functions
- **POST /get_functions/**: Retrieve relevant functions
- **POST /add_doc/**: Add documents to the knowledge base
- **POST /search_doc/**: Search through documents
- **POST /semantic_search_html/**: Search web content
- **POST /get_preferences/**: Get user preferences
- **POST /query_plan/**: Generate query execution plans

## Development Setup

### Running Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio

# Set up environment variables for testing
export OPENAI_API_KEY=your_test_api_key
export COHERE_API_KEY=your_test_cohere_key
export QDRANT_API_KEY=your_test_qdrant_key
export QDRANT_URL=your_test_qdrant_url

# Run tests
python -m pytest tests/ -v
```

### Code Style

The project follows Python best practices. To maintain code quality:

```bash
# Install development dependencies
pip install black flake8 isort

# Format code
black .
isort .

# Lint code
flake8 .
```

## Docker Deployment

### Build Image

```bash
docker build -t superdappai-agent .
```

### Run Container

```bash
docker run -p 8000:80 \
  -e OPENAI_API_KEY=your_key \
  -e QDRANT_API_KEY=your_key \
  -e QDRANT_URL=your_url \
  -e MONGODB_URL=your_mongodb_url \
  superdappai-agent
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM operations | Yes |
| `QDRANT_API_KEY` | Qdrant API key for vector operations | Yes |
| `QDRANT_URL` | Qdrant service URL | Yes |
| `MONGODB_URL` | MongoDB connection string | Yes |
| `COHERE_API_KEY` | Cohere API key for reranking | No |
| `AWS_ACCESS_KEY_ID` | AWS access key for S3 operations | No |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for S3 operations | No |
| `CONSOLE_KEY` | Admin console access key | No |

### Rate Limiting

The platform includes built-in rate limiting:
- Default: 5 requests per second
- Configurable per component
- Async and sync rate limiters available

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest`
5. Submit a pull request

## Security

Please report security vulnerabilities to our team. See [SECURITY.md](SECURITY.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/SuperDappAI/AI/issues)
- **Discussions**: Join the conversation in [GitHub Discussions](https://github.com/SuperDappAI/AI/discussions)

## Roadmap

- [ ] Enhanced multi-modal support
- [ ] Plugin system for custom functions
- [ ] Advanced memory optimization
- [ ] Real-time collaboration features
- [ ] Enhanced security and access controls

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [LangChain](https://langchain.com/) - LLM orchestration
- [Qdrant](https://qdrant.tech/) - Vector database
- [OpenAI](https://openai.com/) - Language models
- [MongoDB](https://mongodb.com/) - Document database
