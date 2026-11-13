# DocAssist

Simple RAG assistant for answering questions over internal PDF documents using FastAPI, LangGraph, OpenAI, and ChromaDB.

## Features

- **PDF Ingestion**: Extracts text from PDF files, splits it into overlapping chunks, and indexes embeddings in ChromaDB.
- **AI Agent (LangGraph)**: Answers questions with citations, extracts structured policy rules, and summarizes documents.
- **REST API (FastAPI)**: Upload new PDFs and ask questions with conversation memory (`thread_id`).

## Project Structure

```text
src/docassist/
├── agent/      # LangGraph agent, tools, prompts, and file cache
├── api/        # FastAPI app, schemas, and error handlers
├── db/         # OpenAI embeddings and ChromaDB vector store
├── extract/    # Structured data extraction (Pydantic)
├── ingest/     # PDF loading, text chunking, and indexing pipeline
├── rag/        # RAG context formatting and prompts
└── config.py   # Central configuration
scripts/        # CLI utilities (build_index.py, ask.py, chat.py)
tests/          # Unit and API tests (pytest)
```

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   .\venv\Scripts\Activate.ps1

2. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```
   python -m pip install -e .

3. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Run the API Server

```bash
uvicorn docassist.api.main:app --reload
```

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### API Endpoints

- `GET /health` – Health check (`{"status": "ok"}`).
- `POST /upload` – Upload a `.pdf` file to `data/` and index it into ChromaDB.
- `POST /ask` – Ask a question (`{"question": "...", "thread_id": "..."}`).

### CLI Scripts

- `python scripts/build_index.py` – Index default PDF documents into ChromaDB.
- `python scripts/ask.py` – Ask a single question from the command line.
- `python scripts/chat.py` – Interactive terminal chat with the agent.

## Running Tests

```bash
pytest
```
