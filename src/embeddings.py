"""Embedding factory: Ollama (local server) or HuggingFace (local, no server).

Both backends implement the LangChain Embeddings interface, so the vector
store code works unchanged regardless of provider.
"""

import logging

from src import config

logger = logging.getLogger(__name__)


def get_embeddings():
    """Create the configured embedding model. Defaults to Ollama."""
    provider = config.EMBEDDING_PROVIDER
    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.info("Using HuggingFace embeddings: %s", config.HF_EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)

    if provider != "ollama":
        logger.warning(
            "Unknown EMBEDDING_PROVIDER '%s', falling back to ollama", provider
        )

    from langchain_community.embeddings import OllamaEmbeddings

    logger.info("Using Ollama embeddings: %s", config.OLLAMA_EMBEDDING_MODEL)
    return OllamaEmbeddings(
        model=config.OLLAMA_EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
    )
