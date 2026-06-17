# Agentic-AI-personal-Assistant

# 🤖 Agentic AI personal Assistant

An AI-powered personal assistant built with **LangGraph**, **Streamlit**, **PostgreSQL**, and **ChromaDB** that combines Retrieval-Augmented Generation (RAG), long-term memory, and configurable tool integrations to deliver personalized, context-aware conversations.

The chatbot can ingest private documents, remember user preferences across sessions, retrieve relevant knowledge from uploaded files, and generate tailored responses using modern Large Language Models (LLMs).

---

## 🚀 Key Highlights

### 🧠 Persistent Memory

* Short-Term Memory (STM) for conversational continuity with PostgreSQL-backend.
* Long-Term Memory for personalized interactions across sessions.

### 📚 Retrieval-Augmented Generation (RAG)

* Upload and query private documents
* Automatic chunking and embedding generation
* Semantic search using vector similarity
* Context-aware answers grounded in retrieved knowledge

### 🔧 Tool-Enabled Architecture

* Built-in Calculator Tool
* Modular tool framework for future extensions
* Feature-level enable/disable support via configuration

### ⚙️ Fully Configuration Driven

All major application behavior can be controlled through:

```text
config/settings.yaml
```

without modifying application code.

### 🤖 Multi-Provider LLM Support

Seamlessly switch between:

* Google Gemini
* OpenAI
* Anthropic
* Groq

### 💾 Flexible Vector Storage

Supports:

* Local ChromaDB
* Chroma Cloud

### 🗄️ Persistent Backend Storage

PostgreSQL is used for:

* Conversation History
* User Memory
* Session Persistence
* Workflow Checkpointing

---

## 🚀 Engineering Highlights

* Implemented Retrieval-Augmented Generation (RAG) using ChromaDB and embedding models.
* Designed a long-term memory system that automatically extracts and stores user-specific information.
* Built configurable workflows using LangGraph.
* Implemented PostgreSQL-backed persistence for conversations, checkpoints, and user memory.
* Added support for multiple LLM providers through a unified configuration layer.
* Designed a feature-toggle architecture allowing capabilities to be enabled or disabled without code changes.
* Containerized the application using Docker for simplified deployment.
* Built a privacy-focused architecture supporting local document storage and vector databases.

---

# 🏗️ Architecture

```text
User
 │
 ▼
Streamlit UI
 │
 ▼
LangGraph Workflow
 │
 ├── Short-Term Memory
 │
 ├── Long-Term Memory (PostgreSQL)
 │
 ├── Tool Execution
 │
 └── RAG Pipeline
       │
       ├── ChromaDB
       ├── Embeddings
       └── Document Retrieval
 │
 ▼
LLM Provider
 │
 ▼
Final Response
```

---

# 🏗️ Engineering Concepts Implemented

- Retrieval-Augmented Generation (RAG)

- Long-Term Memory Systems

- Semantic Search

- Workflow Orchestration

- Feature Toggle Architecture

- Persistent Storage & Checkpointing

- Configuration-Driven System Design

- Containerized Deployment

---

# ⚙️ Configuration

Application behavior is controlled through:

```text
config/settings.yaml
```

---

## LLM Configuration

```yaml
llm:
  provider: google_gemini
  model: gemini-2.5-flash
  temperature: 0.5
  max_tokens: 2048
```

### Supported Providers

| Provider      | Configuration Value |
| ------------- | ------------------- |
| Google Gemini | google_gemini       |
| OpenAI        | openai              |
| Anthropic     | anthropic           |
| Groq          | groq                |

---

## Feature Toggles

Enable or disable application capabilities without changing code.

```yaml
features:
  tools_enabled:
    calculator: true
    memory_enabled: true
```

### Available Features

| Feature        | Description                          |
| -------------- | ------------------------------------ |
| calculator     | Enables calculator tool              |
| memory_enabled | Enables memory and RAG functionality |

Example:

```yaml
memory_enabled: false
```

Disables memory retrieval and document-based context.

---

# 📚 RAG Configuration

## Embedding Model

```yaml
embedding:
  provider: huggingface
  model: sentence-transformers/all-MiniLM-L6-v2
```

Supported Providers:

* HuggingFace
* OpenAI
* Anthropic
* Google

---

## Vector Database

### Local ChromaDB

```yaml
vector_db:
  usage: local

  local:
    persist_directory: chroma_db
```

### Chroma Cloud

```yaml
vector_db:
  usage: cloud

  cloud:
    database: Chroma_db_test
```

---

## Document Processing

```yaml
uploads_dir: data/uploads/
chunk_size: 300
chunk_overlap: 50
top_k: 5
```

| Parameter     | Purpose                    |
| ------------- | -------------------------- |
| uploads_dir   | Uploaded file storage      |
| chunk_size    | Size of generated chunks   |
| chunk_overlap | Chunk overlap size         |
| top_k         | Number of retrieved chunks |

---

# 🧠 Memory System

## Short-Term Memory

```yaml
stm:
  messages_kept: 10
```

Maintains recent conversational context for coherent interactions.

## Long-Term Memory

Automatically extracts and stores:

* User preferences
* Personal details explicitly provided by the user
* Ongoing projects
* Frequently referenced topics

This enables personalized interactions across sessions.

---

# 🔐 Environment Variables

Create a `.env` file using the provided template:

```bash
cp .env.example .env
```

Update the values according to your environment.

Required credentials include:

* LLM API Key
* PostgreSQL Configuration
* Embedding Model API Key (if applicable)
* Chroma Cloud Credentials (optional)

---

# 🚀 Running the Application

## Local Setup

### Prerequisites

* Python 3.11+
* PostgreSQL
* Git

### Clone Repository

```bash
git clone <repository-url>
cd chatbot
```

### Create Virtual Environment

Linux / Mac

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
```

### Configure Application

Update:

```text
config/settings.yaml
```

according to your requirements.

### Start Application

```bash
streamlit run app.py
```

The application automatically initializes the required PostgreSQL database objects on startup.

Open:

```text
http://localhost:8501
```

---

## Docker Setup

### Pull Docker Image

```bash
docker pull <docker-image-name>
```

### Configure Environment

```bash
cp .env.example .env
```

### Run Container

```bash
docker run -d \
  --name personal-chatbot \
  --env-file .env \
  -p 8501:8501 \
  <docker-image-name>
```

Open:

```text
http://localhost:8501
```

---

# 📂 Storage Components

| Component  | Purpose                               |
| ---------- | ------------------------------------- |
| PostgreSQL | Conversations, memories, checkpoints  |
| ChromaDB   | Vector embeddings and semantic search |
| uploads/   | Uploaded document storage             |

---

# 🛠️ Technology Stack

### AI & Orchestration

* LangGraph
* LangChain
* Google Gemini
* OpenAI
* Anthropic
* Groq

### Retrieval & Memory

* ChromaDB
* Sentence Transformers
* PostgreSQL

### Backend

* Python 3.11
* SQLAlchemy
* Psycopg

### Frontend

* Streamlit

### Deployment

* Docker

---

# 🔒 Privacy

This application is designed for processing private documents and personal knowledge bases.

When configured with local storage:

* Documents remain on your infrastructure
* Vector embeddings remain local
* Conversation history remains local
* User memory remains local

Only the configured LLM and embedding providers receive the information required for model inference.

---

# 📄 License

This project is intended for personal and educational use. Feel free to modify and extend it according to your requirements.

