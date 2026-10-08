"""Unit tests for the pluggable LLM and embedding backends (no network required)."""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reload_config(env: dict):
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


def test_llm_provider_defaults_to_ollama():
    _reload_config({})
    import src.llm as llm_module

    importlib.reload(llm_module)
    with patch(
        "langchain_community.llms.Ollama", return_value=MagicMock()
    ) as mock_ollama:
        llm_module.get_llm()
        mock_ollama.assert_called_once()


def test_llm_provider_groq_requires_api_key():
    _reload_config({"LLM_PROVIDER": "groq"})
    import src.llm as llm_module

    importlib.reload(llm_module)
    try:
        llm_module.get_llm()
    except ValueError as e:
        assert "GROQ_API_KEY" in str(e)
    else:
        raise AssertionError("expected ValueError for missing GROQ_API_KEY")


def test_llm_provider_unknown_falls_back_to_ollama():
    _reload_config({"LLM_PROVIDER": "watson"})
    import src.llm as llm_module

    importlib.reload(llm_module)
    with patch(
        "langchain_community.llms.Ollama", return_value=MagicMock()
    ) as mock_ollama:
        llm_module.get_llm()
        mock_ollama.assert_called_once()


def test_embedding_provider_defaults_to_ollama():
    _reload_config({})
    import src.embeddings as emb_module

    importlib.reload(emb_module)
    with patch(
        "langchain_community.embeddings.OllamaEmbeddings",
        return_value=MagicMock(),
    ) as mock_emb:
        emb_module.get_embeddings()
        mock_emb.assert_called_once()


def test_embedding_provider_huggingface():
    _reload_config({"EMBEDDING_PROVIDER": "huggingface"})
    import src.embeddings as emb_module

    importlib.reload(emb_module)
    with patch(
        "langchain_community.embeddings.HuggingFaceEmbeddings",
        return_value=MagicMock(),
    ) as mock_emb:
        emb_module.get_embeddings()
        mock_emb.assert_called_once()
