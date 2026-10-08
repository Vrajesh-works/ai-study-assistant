"""RAG pipeline: retrieval-augmented Q&A, summarization, and definition extraction."""

import logging

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src import config
from src.llm import get_llm
from src.prompts import (
    DEFINITION_EXTRACTION_PROMPT,
    QA_PROMPT_TEMPLATE,
    SUMMARIZATION_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 4000


class RAGSystem:
    """RAG-based Q&A system (LLM backend configured via src.config)."""

    def __init__(self, vector_store: Chroma, temperature: float = 0.3):
        self.vector_store = vector_store

        logger.info("Initializing RAG system")
        self.llm = get_llm(temperature=temperature, max_tokens=512)
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.DEFAULT_RETRIEVAL_K},
        )
        logger.info("RAG system ready")

    def _format_sources(self, docs: list[Document]) -> list[dict]:
        return [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "excerpt": doc.page_content[:200] + "...",
            }
            for doc in docs
        ]

    def ask_question(self, question: str, k: int = 5) -> dict:
        """Answer a question using retrieval-augmented generation."""
        logger.info("Q&A request: %s", question)
        relevant_docs = self.vector_store.similarity_search(question, k=k)

        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information in your study materials.",
                "sources": [],
            }
        logger.info("Found %d relevant chunks", len(relevant_docs))

        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}, "
            f"Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in relevant_docs
        )
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.llm.invoke(prompt)
        logger.info("Answer generated")

        return {"answer": answer, "sources": self._format_sources(relevant_docs)}

    def summarize(
        self, query: str | None = None, summary_type: str = "bullets", k: int = 10
    ) -> dict:
        """Summarize content from the knowledge base."""
        logger.info("Summarization request: type=%s topic=%s", summary_type, query)
        if query:
            relevant_docs = self.vector_store.similarity_search(query, k=k)
        else:
            relevant_docs = self.vector_store.similarity_search(
                "overview main concepts key topics", k=k
            )

        if not relevant_docs:
            return {"summary": "No content found to summarize.", "sources": []}
        logger.info("Found %d relevant chunks", len(relevant_docs))

        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(
            context=context[:MAX_CONTEXT_CHARS], summary_type=summary_type
        )
        summary = self.llm.invoke(prompt)
        logger.info("Summary generated")

        sources = list({doc.metadata.get("source", "unknown") for doc in relevant_docs})
        return {"summary": summary, "sources": sources}

    def extract_definitions(
        self, query: str = "definitions terms concepts", k: int = 10
    ) -> dict:
        """Extract key definitions from the knowledge base."""
        logger.info("Definition extraction request")
        relevant_docs = self.vector_store.similarity_search(query, k=k)

        if not relevant_docs:
            return {"definitions": "No definitions found.", "sources": []}
        logger.info("Found %d relevant chunks", len(relevant_docs))

        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        prompt = DEFINITION_EXTRACTION_PROMPT.format(
            context=context[:MAX_CONTEXT_CHARS]
        )
        definitions = self.llm.invoke(prompt)
        logger.info("Definitions extracted")

        sources = list({doc.metadata.get("source", "unknown") for doc in relevant_docs})
        return {"definitions": definitions, "sources": sources}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("RAG System Module - Ready!")
    print("\nTo test:")
    print("1. First create a vector store using ingestion.py")
    print("2. Then run:")
    print("\n  from src.ingestion import DocumentIngestion")
    print("  from src.rag import RAGSystem")
    print("  ingestion = DocumentIngestion()")
    print("  vector_store = ingestion.load_vector_store('test_store')")
    print("  rag = RAGSystem(vector_store)")
    print("  result = rag.ask_question('What is machine learning?')")
    print("  print(result['answer'])")
