"""LLM factory: Ollama (local) or Groq (hosted API).

Both backends expose a minimal `.invoke(prompt) -> str` interface so the
RAG pipeline and quiz generator work unchanged regardless of provider.
"""

import logging

from src import config

logger = logging.getLogger(__name__)


class _GroqStringWrapper:
    """Adapt langchain-groq's chat model to the string-in/string-out interface."""

    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=groq requires the GROQ_API_KEY environment variable. "
                "Get a free key at https://console.groq.com"
            )

        from langchain_groq import ChatGroq

        self._llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=config.GROQ_API_KEY,
        )
        logger.info("Using Groq LLM: %s", model_name)

    def invoke(self, prompt: str) -> str:
        return self._llm.invoke(prompt).content


def get_llm(temperature: float = 0.3, max_tokens: int = 1024):
    """Create the configured LLM. Defaults to local Ollama."""
    provider = config.LLM_PROVIDER
    if provider == "groq":
        return _GroqStringWrapper(
            model_name=config.GROQ_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider != "ollama":
        logger.warning("Unknown LLM_PROVIDER '%s', falling back to ollama", provider)

    from langchain_community.llms import Ollama

    logger.info("Using Ollama LLM: %s", config.OLLAMA_MODEL)
    return Ollama(
        model=config.OLLAMA_MODEL,
        temperature=temperature,
        base_url=config.OLLAMA_BASE_URL,
        num_predict=max_tokens,
    )
