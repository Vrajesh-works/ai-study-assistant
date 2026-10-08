"""Centralized configuration.

Values are read from the environment, with an optional `.env` file in the
project root (see `.env.example`). This keeps hostnames, ports, and model
names out of the source code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _getenv(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


# LLM provider: "ollama" (local) or "groq" (hosted API)
LLM_PROVIDER: str = _getenv("LLM_PROVIDER", "ollama").lower()

# Ollama
OLLAMA_BASE_URL: str = _getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = _getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL: str = _getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Groq (hosted LLM API, free tier at https://console.groq.com)
GROQ_API_KEY: str = _getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = _getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Embedding provider: "ollama" (local) or "huggingface" (local, no server needed)
EMBEDDING_PROVIDER: str = _getenv("EMBEDDING_PROVIDER", "ollama").lower()
HF_EMBEDDING_MODEL: str = _getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# API server
API_HOST: str = _getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(_getenv("API_PORT", "8000"))

# RAG defaults
DEFAULT_RETRIEVAL_K: int = int(_getenv("DEFAULT_RETRIEVAL_K", "5"))
CHUNK_SIZE: int = int(_getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(_getenv("CHUNK_OVERLAP", "200"))

# Storage
VECTOR_STORE_DIR: str = _getenv("VECTOR_STORE_DIR", "data/vector_store")
UPLOAD_DIR: str = _getenv("UPLOAD_DIR", "data/uploads")
VECTOR_STORE_NAME: str = _getenv("VECTOR_STORE_NAME", "study_materials")
