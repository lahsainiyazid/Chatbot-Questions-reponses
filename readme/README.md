# Arabic RAG Chatbot for Moroccan Administrative Services

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Moroccan administrative procedures using official government documents.

The system combines hybrid retrieval, reranking, and a large language model to provide accurate and context-aware answers in Arabic.

---

## Features

- Hybrid retrieval (Dense + BM25)
- Chroma vector database
- Cross-encoder reranking
- FastAPI REST API
- Arabic question answering
- Modular project structure
- Response time monitoring
- Token usage tracking

---

## Tech Stack

### Backend

- Python
- FastAPI
- Uvicorn

### Retrieval

- LangChain
- ChromaDB
- BM25 Retriever
- Ensemble Retriever

### Embeddings

- intfloat/multilingual-e5-large

### Reranker

- cross-encoder/ms-marco-MiniLM-L6-v2

### Language Model

- Llama 3.3 70B (via Groq API)

### Evaluation

- RAGAS

---

## Project Structure

```
project/
│
├── api/
│   ├── main.py
│   └── routes.py
│
├── retrieval/
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│
├── data/
│
├── database/
│
├── evaluation/
│
├── config.py
│
├── requirements.txt
│
└── README.md
```

---

## Pipeline

```
User Question
      │
      ▼
Hybrid Retrieval
(Dense + BM25)
      │
      ▼
Top-k Documents
      │
      ▼
Cross Encoder Reranker
      │
      ▼
Relevant Context
      │
      ▼
LLM
      │
      ▼
Final Answer
```

---

## API

### POST `/ask`

Request

```json
{
    "question": "ما هي الوثائق المطلوبة لإنشاء مقاولة؟"
}
```

Response

```json
{
    "question": "...",
    "answer": "...",
    "retrieval_time": 0.21,
    "reranker_time": 0.84,
    "llm_time": 0.56,
    "total_time": 1.61,
    "token_usage": {
        ...
    }
}
```

---

## Retrieval Strategy

The chatbot uses a hybrid retrieval pipeline:

- Dense semantic search using multilingual embeddings
- Sparse lexical search using BM25
- Ensemble retrieval
- Cross-encoder reranking to improve document relevance before generation

---

## Evaluation

The system is evaluated using RAGAS metrics:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Latency metrics are also collected for:

- Retrieval
- Reranking
- LLM generation
- Total response time

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/your-project.git

cd your-project
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file.

Example

```text
GROQ_API_KEY=your_api_key
```

---

## Running the API

```bash
uvicorn api.main:app --reload
```

The API will be available at

```
http://127.0.0.1:8000
```

Interactive documentation

```
http://127.0.0.1:8000/docs
```

---

## Future Improvements

- SQL document storage
- Docker deployment
- Query rewriting
- Streaming responses
- User authentication
- Caching
- Admin dashboard
- Conversation memory
- Automated evaluation pipeline

---

## Disclaimer

This project is intended for educational and research purposes. Answers are generated from the indexed document collection and should be verified against official government sources when used for administrative decisions.
