# 🎓 AI-Powered Study Assistant

> Transform your study materials into an intelligent learning companion using RAG (Retrieval Augmented Generation) and local LLMs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-black)](https://ollama.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io)

---

## 📋 Overview

An intelligent study assistant that uses **Retrieval Augmented Generation (RAG)** to help students learn more effectively. Upload your textbooks, lecture notes, or study materials and get:

- 💬 **Intelligent Q&A** - Ask questions and get accurate answers with source citations
- 📝 **Smart Summaries** - Generate concise summaries in multiple formats
- 🎯 **Auto-Generated Quizzes** - Create custom quizzes with automatic grading
- 📖 **Definition Extraction** - Extract and organize key terms and concepts

**🆓 100% FREE** - Runs completely locally using Ollama. No API costs, no rate limits, no cloud dependencies.

---

## 🎬 Demo

### Frontend Interface

![AI Study Assistant Interface](docs/images/frontend-screenshot.png)


## ✨ Features

### Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| 📄 **Document Processing** | PDF & TXT file support with intelligent chunking | ✅ |
| 🔍 **Semantic Search** | Vector-based similarity search using Chroma DB | ✅ |
| 💬 **RAG Q&A** | Context-aware answers with source citations | ✅ |
| 📝 **Multi-Format Summaries** | Bullets, short, detailed, and ELI15 formats | ✅ |
| 🎯 **Quiz Generation** | Multiple choice questions with 3 difficulty levels | ✅ |
| ✅ **Auto-Grading** | Instant quiz grading with explanations | ✅ |
| 📖 **Definition Extraction** | Automatic key term identification | ✅ |
| 🌐 **REST API** | Full FastAPI backend with Swagger docs | ✅ |
| 🖥️ **Web Interface** | Beautiful Streamlit UI | ✅ |
| 🔒 **100% Local** | No data leaves your computer | ✅ |

### Advanced Features

- **Multiple Summary Types**: Bullets, short summaries, detailed explanations, or simplified (ELI15) versions
- **Difficulty Levels**: Easy, medium, and hard quiz questions
- **Source Citations**: Every answer includes references to source documents
- **Batch Processing**: Upload and index multiple documents
- **Real-time Processing**: Live feedback and progress indicators

---

## 🏗️ Architecture
```
┌─────────────────┐
│   Streamlit UI  │  ← User Interface
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   FastAPI       │  ← REST API Layer
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────────┐
│         RAG Pipeline                │
│  ┌──────────┐  ┌──────────────┐   │
│  │ Ingestion│→ │ Vector Store │   │
│  └──────────┘  └──────────────┘   │
│       ↓              ↓              │
│  ┌──────────────────────────────┐ │
│  │    RAG System (Q&A)          │ │
│  └──────────────────────────────┘ │
│  ┌──────────────────────────────┐ │
│  │    Quiz Generator            │ │
│  └──────────────────────────────┘ │
└─────────────────────────────────────┘
         │
         ↓
┌─────────────────┐
│  Ollama (Local) │  ← LLM & Embeddings
│  - llama3.2     │
│  - nomic-embed  │
└─────────────────┘
```

### Technology Stack

- **LLM Framework**: LangChain 0.1.20
- **Vector Database**: ChromaDB 0.4.24
- **Embeddings**: Ollama (nomic-embed-text)
- **Language Model**: Ollama (llama3.2 / llama3.1 / mistral)
- **Backend**: FastAPI 0.109.0
- **Frontend**: Streamlit 1.32.0
- **Document Processing**: PyPDF 4.0.1

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** - [Download here](https://www.python.org/downloads/)
- **Ollama** - [Install from ollama.ai](https://ollama.ai/download)
- **8GB+ RAM** recommended for optimal performance

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/Vrajesh-works/ai-study-assistant.git
   cd ai-study-assistant
```

2. **Install Ollama and download models**
```bash
   # Install Ollama from https://ollama.ai/download
   
   # Download required models
   ollama pull llama3.2          # or llama3.1, mistral
   ollama pull nomic-embed-text  # for embeddings
```

3. **Create virtual environment**
```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
```

4. **Install dependencies**
```bash
   pip install -r requirements.txt
```

5. **Run the application**

   **Terminal 1 - Start API:**
```bash
   python api/main.py
```
   
   **Terminal 2 - Start Frontend:**
```bash
   streamlit run frontend/app.py
```

6. **Open your browser**
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs

---

## 📖 Usage Guide

### 1. Upload Documents

- Click **"Choose a file"** in the sidebar
- Select PDF or TXT files (study notes, textbooks, lecture slides)
- Click **"📤 Process Document"**
- Wait for processing (30-60 seconds for first upload)

### 2. Ask Questions

Navigate to **"💬 Ask Questions"** tab:
```
Question: "What is machine learning?"
↓
AI searches your documents
↓
Returns answer with source citations
```

**Example Questions:**
- "Explain gradient descent with an example"
- "What are the types of machine learning?"
- "Compare supervised and unsupervised learning"

### 3. Generate Summaries

Navigate to **"📝 Summarize"** tab:

- **Topic**: Specify a topic or leave empty for general summary
- **Type**: Choose from bullets, short, detailed, or eli15
- Get organized summaries with source references

### 4. Take Quizzes

Navigate to **"🎯 Quiz Me"** tab:

1. Enter a topic (e.g., "Chapter 2", "Neural Networks")
2. Choose number of questions (3-10)
3. Select difficulty (easy/medium/hard)
4. Click **"Generate Quiz"**
5. Answer questions and submit for instant grading

### 5. Extract Definitions

Navigate to **"📚 Definitions"** tab:

- Extract key terms and definitions automatically
- Filter by topic or get all definitions
- Perfect for creating glossaries or flashcards

---

## 📁 Project Structure
```
ai-study-assistant/
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI application
├── frontend/
│   └── app.py                  # Streamlit interface
├── src/
│   ├── __init__.py
│   ├── ingestion.py            # Document processing
│   ├── rag.py                  # RAG Q&A system
│   ├── quiz_generator.py       # Quiz creation & grading
│   └── prompts.py              # Prompt templates
├── data/
│   ├── uploads/                # Uploaded documents
│   └── vector_store/           # Vector embeddings
├── tests/
│   └── test_system.py          # System tests
├── docs/
│   └── images/                 # Screenshots
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (optional)
└── README.md                   # This file
```

---

## 🔧 Configuration

### Ollama Models

Default models can be changed in the source files:

**For LLM (Text Generation):**
```python
# In src/rag.py and src/quiz_generator.py
model_name = "llama3.2"  # Options: llama3.2, llama3.1, mistral, phi
```

**For Embeddings:**
```python
# In src/ingestion.py
model = "nomic-embed-text"  # Recommended for best results
```

### Environment Variables

Create `.env` file (optional):
```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

---

## 📊 Performance Benchmarks

| Operation | First Run | Subsequent Runs |
|-----------|-----------|-----------------|
| Document Upload (10 pages) | 30-60s | 20-40s |
| Question Answering | 5-15s | 5-15s |
| Summary Generation | 10-20s | 10-20s |
| Quiz Generation (3 questions) | 20-40s | 20-40s |
| Definition Extraction | 10-20s | 10-20s |

*Tested on: Intel i5 / 16GB RAM / No GPU*

**Performance Tips:**
- First embedding generation is slower (building index)
- Use smaller models (phi, mistral) for faster responses
- Chunk size affects accuracy vs speed trade-off

---

## 🧪 Testing

Run the complete system test:
```bash
python tests/test_system.py
```

This will:
1. Create a test document
2. Process and index it
3. Test Q&A functionality
4. Test summarization
5. Generate and grade a quiz
6. Extract definitions

Expected output: `✅ ALL TESTS COMPLETED SUCCESSFULLY!`

---

## 🐛 Troubleshooting

<details>
<summary><b>API shows "No module named 'src'"</b></summary>

**Solution:**
```bash
# Run from project root
cd ai-study-assistant
python api/main.py
```

Or add to top of `api/main.py`:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
</details>

<details>
<summary><b>Streamlit shows "API not reachable"</b></summary>

**Check:**
1. API is running on port 8000
2. Visit http://localhost:8000 to verify
3. Check for port conflicts
4. Restart API if needed
</details>

<details>
<summary><b>Ollama connection failed</b></summary>

**Solution:**
```bash
# Check Ollama is running
ollama list

# Start Ollama service (if needed)
ollama serve

# Test model
ollama run llama3.2
```
</details>

<details>
<summary><b>Quiz generation fails or returns invalid JSON</b></summary>

**Solutions:**
- Reduce number of questions to 3-5
- Try "easy" difficulty first
- Use more specific topics
- Check if document has enough content
- Try a different model (mistral, llama3.1)
</details>

<details>
<summary><b>Slow performance</b></summary>

**Optimizations:**
- Use smaller model: `ollama pull phi`
- Reduce chunk size in `src/ingestion.py`
- Lower `k` value (fewer chunks retrieved)
- Use SSD for vector store
- Close other applications
</details>



</div>
