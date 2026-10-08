"""FastAPI backend for the AI Study Assistant.

Run from the project root:
    python api/main.py
"""

import logging
import os
import shutil
import sys
import urllib.request
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Fix import path so `src` resolves when running api/main.py directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src import config
from src.ingestion import SUPPORTED_EXTENSIONS, DocumentIngestion
from src.quiz_generator import QuizGenerator
from src.rag import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(parent_dir, config.UPLOAD_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="AI Study Assistant API",
    description=(
        "RAG-powered study assistant. Upload study materials, ask questions, "
        "generate summaries and quizzes. Runs on local LLMs via Ollama."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class SummaryType(str, Enum):
    bullets = "bullets"
    short = "short"
    detailed = "detailed"
    eli5 = "eli5"


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class SummarizeRequest(BaseModel):
    topic: str | None = None
    summary_type: SummaryType = SummaryType.bullets
    k: int = Field(default=10, ge=1, le=20)


class DefinitionsRequest(BaseModel):
    topic: str = "definitions terms concepts"
    k: int = Field(default=10, ge=1, le=20)


class QuizRequest(BaseModel):
    topic: str = Field(min_length=1)
    num_questions: int = Field(default=5, ge=1, le=10)
    difficulty: Difficulty = Difficulty.medium


class GradeQuizRequest(BaseModel):
    questions: list[dict]
    user_answers: dict[str, str]


class _State:
    """Lazily-initialized pipeline components (built on first upload)."""

    def __init__(self):
        self.ingestion: DocumentIngestion | None = None
        self.vector_store = None
        self.rag_system: RAGSystem | None = None
        self.quiz_generator: QuizGenerator | None = None

    def ensure_ingestion(self) -> DocumentIngestion:
        if self.ingestion is None:
            self.ingestion = DocumentIngestion()
        return self.ingestion

    def require_ready(self):
        if self.rag_system is None or self.quiz_generator is None:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded yet. Please upload documents first.",
            )


state = _State()


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(
            f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3
        ) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - any failure means unreachable
        return False


@app.get("/", tags=["status"])
def root():
    """Service status and endpoint index."""
    return {
        "message": "AI Study Assistant API",
        "status": "running",
        "documents_loaded": state.vector_store is not None,
        "ollama": {
            "base_url": config.OLLAMA_BASE_URL,
            "model": config.OLLAMA_MODEL,
            "reachable": _ollama_reachable(),
        },
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "ask": "/ask",
            "summarize": "/summarize",
            "definitions": "/definitions",
            "quiz_generate": "/quiz/generate",
            "quiz_grade": "/quiz/grade",
            "documents": "/documents",
        },
    }


@app.get("/health", tags=["status"])
def health():
    """Health check, including Ollama connectivity."""
    ollama_ok = _ollama_reachable()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "documents_loaded": state.vector_store is not None,
        "ollama_reachable": ollama_ok,
        "ollama_base_url": config.OLLAMA_BASE_URL,
    }


@app.post("/upload", tags=["documents"])
async def upload_document(file: UploadFile = File(...)):  # noqa: B008 - standard FastAPI
    """Upload a PDF or TXT file, chunk it, and index it in the vector store."""
    global_state = state
    logger.info("Upload request: %s (%s)", file.filename, file.content_type)

    file_extension = Path(file.filename or "").suffix.lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    file_path = os.path.join(UPLOAD_DIR, Path(file.filename).name)
    content = await file.read()
    # Small write; kept inline rather than pushed to a threadpool.
    with open(file_path, "wb") as buffer:  # noqa: ASYNC230
        buffer.write(content)
    logger.info("Saved upload to %s", file_path)

    try:
        ingestion = global_state.ensure_ingestion()
        chunks = ingestion.process_documents(file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Document processing failed")
        raise HTTPException(status_code=500, detail="Failed to process document")

    if not chunks:
        raise HTTPException(status_code=400, detail="No content extracted from document")

    try:
        if global_state.vector_store is None:
            logger.info("Creating new vector store")
            global_state.vector_store = ingestion.create_vector_store(
                chunks, config.VECTOR_STORE_NAME
            )
        else:
            logger.info("Appending to existing vector store")
            ingestion.add_documents_to_existing_store(chunks, config.VECTOR_STORE_NAME)
            global_state.vector_store = ingestion.load_vector_store(
                config.VECTOR_STORE_NAME
            )
    except Exception:
        logger.exception("Vector store error")
        raise HTTPException(status_code=500, detail="Failed to index document")

    global_state.rag_system = RAGSystem(global_state.vector_store)
    global_state.quiz_generator = QuizGenerator(global_state.vector_store)
    logger.info("Upload complete: %d chunks", len(chunks))

    return {
        "message": "Document uploaded and processed successfully",
        "filename": Path(file.filename).name,
        "chunks_created": len(chunks),
        "status": "success",
    }


@app.post("/ask", tags=["qa"])
def ask_question(request: QuestionRequest):
    """Answer a question from the uploaded study materials."""
    state.require_ready()
    try:
        result = state.rag_system.ask_question(request.question, k=request.k)
        logger.info("Answered question with %d sources", len(result["sources"]))
        return result
    except Exception:
        logger.exception("Q&A failed")
        raise HTTPException(status_code=500, detail="Question answering failed")


@app.post("/summarize", tags=["qa"])
def summarize(request: SummarizeRequest):
    """Summarize the uploaded study materials."""
    state.require_ready()
    try:
        result = state.rag_system.summarize(
            query=request.topic,
            summary_type=request.summary_type.value,
            k=request.k,
        )
        logger.info("Summary generated from %d sources", len(result["sources"]))
        return result
    except Exception:
        logger.exception("Summarization failed")
        raise HTTPException(status_code=500, detail="Summarization failed")


@app.post("/definitions", tags=["qa"])
def get_definitions(request: DefinitionsRequest):
    """Extract key terms and definitions from the uploaded materials."""
    state.require_ready()
    try:
        result = state.rag_system.extract_definitions(query=request.topic, k=request.k)
        logger.info("Definitions extracted from %d sources", len(result["sources"]))
        return result
    except Exception:
        logger.exception("Definition extraction failed")
        raise HTTPException(status_code=500, detail="Definition extraction failed")


@app.post("/quiz/generate", tags=["quiz"])
def generate_quiz(request: QuizRequest):
    """Generate a multiple-choice quiz from the uploaded materials."""
    state.require_ready()
    try:
        quiz = state.quiz_generator.generate_quiz(
            topic=request.topic,
            num_questions=request.num_questions,
            difficulty=request.difficulty.value,
        )
        if "error" in quiz:
            raise HTTPException(status_code=500, detail=quiz["error"])
        logger.info("Generated %d quiz questions", len(quiz.get("questions", [])))
        return quiz
    except HTTPException:
        raise
    except Exception:
        logger.exception("Quiz generation failed")
        raise HTTPException(status_code=500, detail="Quiz generation failed")


@app.post("/quiz/grade", tags=["quiz"])
def grade_quiz(request: GradeQuizRequest):
    """Grade a submitted quiz."""
    state.require_ready()
    try:
        # Normalize answer keys to int for the grader
        user_answers = {int(k): v for k, v in request.user_answers.items()}
        results = state.quiz_generator.grade_quiz(request.questions, user_answers)
        logger.info("Quiz graded: %s%%", results["score"])
        return results
    except Exception:
        logger.exception("Quiz grading failed")
        raise HTTPException(status_code=500, detail="Quiz grading failed")


@app.get("/documents", tags=["documents"])
def list_documents():
    """List uploaded documents."""
    try:
        files = [
            f
            for f in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, f))
        ]
        return {"documents": files, "count": len(files)}
    except Exception:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail="Failed to list documents")


@app.delete("/reset", tags=["documents"])
def reset_system():
    """Delete all uploaded documents and the vector store."""
    try:
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        vector_store_path = os.path.join(
            parent_dir, config.VECTOR_STORE_DIR, config.VECTOR_STORE_NAME
        )
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)

        state.vector_store = None
        state.rag_system = None
        state.quiz_generator = None
        logger.info("System reset")
        return {"message": "System reset successfully", "status": "success"}
    except Exception:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail="Reset failed")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting AI Study Assistant API on %s:%d", config.API_HOST, config.API_PORT)
    logger.info("API docs: http://localhost:%d/docs", config.API_PORT)
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="info")
