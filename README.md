# SuperDappAI

AI agent platform with memory management, document processing, and function orchestration.

## Features

- Long-term memory with conversation summarization
- Document ingestion and semantic search  
- Dynamic function discovery and execution
- Web content processing and analysis
- Rate limiting and caching
- User preferences management

## Components

- `AgentManager` - Memory and conversation handling
- `FunctionsManager` - Function discovery and execution
- `DocumentManager` - Document processing and search
- `WebManager` - Web content analysis
- `MemorySummarizer` - Conversation optimization
- `PreferencesResolver` - User preference storage

## Requirements

- Python 3.11+
- OpenAI API key
- Qdrant vector database
- MongoDB instance

## Installation

```bash
git clone https://github.com/SuperDappAI/AI.git
cd AI
pip install -r requirements.txt
```

## Configuration

Copy and edit the environment file:

```bash
cp .env.example .env
```

Required environment variables:

```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your_qdrant_instance.com
MONGODB_URL=mongodb://localhost:27017/preferences
```

## Usage

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

API documentation available at `http://localhost:8000/docs`

## API Endpoints

- `POST /push_memory/` - Store memories
- `POST /pull_memory/` - Retrieve memories  
- `POST /push_functions/` - Register functions
- `POST /get_functions/` - Query functions
- `POST /add_doc/` - Add documents
- `POST /search_doc/` - Search documents
- `POST /semantic_search_html/` - Search web content
- `POST /get_preferences/` - Get user preferences

## Development

Run tests:

```bash
python -m pytest tests/
```

Format code:

```bash
black .
isort .
```

## Docker

```bash
docker build -t superdappai .
docker run -p 8000:80 --env-file .env superdappai
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.
