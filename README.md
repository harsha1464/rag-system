# AI-Powered RAG Backend System

A production-style Retrieval-Augmented Generation (RAG) backend built using FastAPI, FAISS, Sentence Transformers, and Ollama.
This project allows users to upload PDF documents, perform semantic search on document chunks, and generate grounded AI responses using a local LLM.

---

# Features

* PDF document ingestion
* Text extraction using pdfplumber
* Intelligent chunking with overlap
* Semantic embeddings using Sentence Transformers
* Vector similarity search using FAISS
* Local LLM inference using Ollama + Mistral
* REST APIs with FastAPI
* Swagger API documentation
* Persistent local vector storage

---

# Tech Stack

## Backend

* Python 3.11
* FastAPI
* Uvicorn

## AI / RAG

* Sentence Transformers
* FAISS
* Ollama
* Mistral Model

## Document Processing

* pdfplumber
* NumPy
* Pickle

---

# Project Architecture

```text
PDF Upload
    ↓
Text Extraction (pdfplumber)
    ↓
Chunking
    ↓
Sentence Embeddings
    ↓
FAISS Vector Index
    ↓
Semantic Retrieval
    ↓
Prompt Construction
    ↓
Ollama (Mistral)
    ↓
Grounded AI Response
```

---

# API Endpoints

## 1. Upload PDF

```http
POST /ingest/pdf
```

Uploads and indexes PDF documents.

---

## 2. Semantic Search

```http
POST /search
```

Performs vector similarity search on indexed chunks.

---

## 3. RAG Query

```http
POST /rag/query
```

Retrieves relevant chunks and generates grounded AI responses using Ollama.

---

# Local Setup

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd rag-system
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate environment:

```bash
.\.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download and install:

[Ollama Official Website](https://ollama.com/?utm_source=chatgpt.com)

---

# Pull Mistral Model

```bash
ollama pull mistral
```

---

# Verify Ollama

```bash
ollama list
```

---

# Start Ollama Server

```bash
ollama serve
```

If port already exists, Ollama is already running.

---

# Run Backend

Navigate to backend folder:

```bash
cd backend
```

Activate virtual environment:

```bash
.\.venv\Scripts\activate
```

Run FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

---

# Swagger API Docs

Open:

```text
http://127.0.0.1:8000/docs
```

---

# How Retrieval Works

1. PDF text is extracted and chunked
2. Each chunk is converted into a 384-dimensional embedding
3. Embeddings are stored inside a FAISS vector index
4. User query is embedded using the same model
5. FAISS retrieves top-k semantically similar chunks
6. Retrieved chunks are added into a prompt template
7. Prompt is sent to local Ollama Mistral model
8. Grounded AI response is generated

---

# Embedding Model

Model Used:

```text
all-MiniLM-L6-v2
```

* Lightweight
* Fast inference
* 384-dimensional embeddings
* Optimized for semantic similarity tasks

---

# Vector Database

FAISS is used as a local vector index.

Stored Files:

```text
faiss.index   -> vector embeddings
chunks.pkl    -> actual chunk text
```

FAISS stores vectors and positional IDs, while chunk mapping is maintained separately using Python lists.


# Author

Sriharsha Bandaru

Built as an end-to-end AI Engineering + RAG Systems project for learning production-grade LLM application development.
