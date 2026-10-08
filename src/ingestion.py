"""Document ingestion: extract text from PDF/TXT, chunk it, and index it in Chroma."""

import json
import logging
import os
from pathlib import Path

import pypdf
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentIngestion:
    """Handles document upload, processing, and indexing with Ollama embeddings."""

    def __init__(
        self,
        vector_store_path: str = config.VECTOR_STORE_DIR,
        embedding_model: str = config.OLLAMA_EMBEDDING_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
    ):
        self.vector_store_path = vector_store_path

        logger.info("Initializing Ollama embeddings (model=%s)", embedding_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def extract_text_from_pdf(self, pdf_path: str) -> list[dict]:
        """Extract text from a PDF, keeping page numbers in metadata."""
        documents = []
        try:
            logger.info("Extracting text from %s", pdf_path)
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                filename = Path(pdf_path).name

                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(
                            {
                                "content": text,
                                "metadata": {
                                    "source": filename,
                                    "page": page_num,
                                    "type": "pdf",
                                },
                            }
                        )
            logger.info("Extracted %d pages from %s", len(documents), filename)
        except Exception:
            logger.exception("Error extracting PDF %s", pdf_path)
        return documents

    def extract_text_from_txt(self, txt_path: str) -> list[dict]:
        """Extract text from a plain text file."""
        try:
            logger.info("Reading text file %s", txt_path)
            with open(txt_path, "r", encoding="utf-8") as file:
                text = file.read()
            filename = Path(txt_path).name
            logger.info("Read %d characters from %s", len(text), filename)
            return [{"content": text, "metadata": {"source": filename, "type": "txt"}}]
        except Exception:
            logger.exception("Error reading text file %s", txt_path)
            return []

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace and strip control characters."""
        text = " ".join(text.split())
        text = text.replace("\x00", "")
        return text.strip()

    def process_documents(self, file_path: str) -> list[Document]:
        """Process a document file into LangChain document chunks."""
        logger.info("Processing: %s", file_path)
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".pdf":
            raw_docs = self.extract_text_from_pdf(file_path)
        elif file_extension == ".txt":
            raw_docs = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        if not raw_docs:
            raise ValueError("No content extracted from document")

        documents = []
        for doc in raw_docs:
            cleaned_text = self.clean_text(doc["content"])
            if cleaned_text and len(cleaned_text) > 50:
                documents.append(
                    Document(page_content=cleaned_text, metadata=doc["metadata"])
                )
        logger.info("Created %d clean documents", len(documents))

        chunks = self.text_splitter.split_documents(documents)
        logger.info("Created %d chunks", len(chunks))
        return chunks

    def create_vector_store(self, documents: list[Document], store_name: str = "default"):
        """Create a Chroma vector store from documents."""
        if not documents:
            raise ValueError("No documents to index")

        store_path = os.path.join(self.vector_store_path, store_name)
        os.makedirs(store_path, exist_ok=True)

        logger.info(
            "Creating vector store '%s' with %d chunks (this may take a few minutes)",
            store_name,
            len(documents),
        )
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=store_path,
            collection_name=store_name,
        )

        metadata = {
            "num_documents": len(documents),
            "sources": list(
                {doc.metadata.get("source", "unknown") for doc in documents}
            ),
            "store_name": store_name,
        }
        with open(os.path.join(store_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Vector store created at %s (%d chunks)", store_path, len(documents))
        return vector_store

    def load_vector_store(self, store_name: str = "default"):
        """Load an existing Chroma vector store."""
        store_path = os.path.join(self.vector_store_path, store_name)
        if not os.path.exists(store_path):
            raise FileNotFoundError(f"Vector store not found at {store_path}")

        logger.info("Loading vector store from %s", store_path)
        vector_store = Chroma(
            persist_directory=store_path,
            embedding_function=self.embeddings,
            collection_name=store_name,
        )
        logger.info("Vector store loaded")
        return vector_store

    def add_documents_to_existing_store(
        self, documents: list[Document], store_name: str = "default"
    ):
        """Append documents to an existing store, creating it if needed."""
        try:
            vector_store = self.load_vector_store(store_name)
            logger.info("Adding %d documents to '%s'", len(documents), store_name)
            vector_store.add_documents(documents)
            logger.info("Documents added successfully")
        except FileNotFoundError:
            logger.info("No existing store found, creating '%s'", store_name)
            vector_store = self.create_vector_store(documents, store_name)
        return vector_store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Document Ingestion Module - Ready!")
    print("\nTo test:")
    print("1. Place a PDF or TXT file in data/uploads/")
    print("2. Run:")
    print("\n  from src.ingestion import DocumentIngestion")
    print("  ingestion = DocumentIngestion()")
    print("  chunks = ingestion.process_documents('data/uploads/your_file.pdf')")
    print("  vector_store = ingestion.create_vector_store(chunks, 'test_store')")
