"""Unit tests for configuration defaults and environment overrides."""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reload_config(env: dict):
    """Reload src.config with a controlled environment."""
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        import src.config as cfg
        importlib.reload(cfg)
        return cfg
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_defaults():
    cfg = _reload_config({})
    assert cfg.OLLAMA_BASE_URL == "http://localhost:11434"
    assert cfg.OLLAMA_MODEL == "llama3.2"
    assert cfg.OLLAMA_EMBEDDING_MODEL == "nomic-embed-text"
    assert cfg.API_PORT == 8000
    assert cfg.API_HOST == "0.0.0.0"
    assert cfg.VECTOR_STORE_NAME == "study_materials"


def test_env_overrides():
    cfg = _reload_config(
        {
            "OLLAMA_BASE_URL": "http://ollama:11434",
            "OLLAMA_MODEL": "mistral",
            "OLLAMA_EMBEDDING_MODEL": "mxbai-embed-large",
            "API_PORT": "9000",
        }
    )
    assert cfg.OLLAMA_BASE_URL == "http://ollama:11434"
    assert cfg.OLLAMA_MODEL == "mistral"
    assert cfg.OLLAMA_EMBEDDING_MODEL == "mxbai-embed-large"
    assert cfg.API_PORT == 9000
