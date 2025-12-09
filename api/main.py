import sys
import os

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import shutil
from pathlib import Path

from src.ingestion import DocumentIngestion
from src.rag import RAGSystem
from src.quiz_generator import QuizGenerator

# Initialize FastAPI
app = FastAPI(
    title="AI Study Assistant API",
    description="RAG-powered study assistant using Ollama (100% FREE)",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
ingestion = DocumentIngestion()
vector_store = None
rag_system = None
quiz_generator = None

UPLOAD_DIR = os.path.join(parent_dir, "data", "uploads")
VECTOR_STORE_NAME = "study_materials"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# Pydantic models for request validation
class QuestionRequest(BaseModel):
    question: str
    k: int = 5


class SummarizeRequest(BaseModel):
    topic: Optional[str] = None
    summary_type: str = "bullets"
    k: int = 10


class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 10
    difficulty: str = "medium"


class GradeQuizRequest(BaseModel):
    questions: List[Dict]
    user_answers: Dict[int, str]


# API Endpoints

@app.get("/")
def root():
    """Health check and status endpoint"""
    return {
        "message": "AI Study Assistant API - Powered by Ollama",
        "status": "running",
        "documents_loaded": vector_store is not None,
        "ollama_url": "http://localhost:11434",
        "endpoints": {
            "upload": "/upload",
            "ask": "/ask",
            "summarize": "/summarize",
            "definitions": "/definitions",
            "quiz_generate": "/quiz/generate",
            "quiz_grade": "/quiz/grade",
            "documents": "/documents"
        }
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF or TXT)"""
    global vector_store, rag_system, quiz_generator
    
    try:
        print(f"\n{'='*60}")
        print(f"FILE UPLOAD REQUEST")
        print(f"{'='*60}")
        print(f"Filename: {file.filename}")
        print(f"Content Type: {file.content_type}")
        
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.pdf', '.txt']:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Only .pdf and .txt are supported."
            )
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        print(f"Saving to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"✓ File saved")
        
        # Process document
        print("Processing document...")
        chunks = ingestion.process_documents(file_path)
        
        if not chunks:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract content from document"
            )
        
        # Create or update vector store
        try:
            if vector_store is None:
                print("Creating new vector store...")
                vector_store = ingestion.create_vector_store(chunks, VECTOR_STORE_NAME)
            else:
                print("Updating existing vector store...")
                ingestion.add_documents_to_existing_store(chunks, VECTOR_STORE_NAME)
                vector_store = ingestion.load_vector_store(VECTOR_STORE_NAME)
        except Exception as e:
            print(f"Vector store error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create/update vector store: {str(e)}"
            )
        
        # Initialize RAG and Quiz systems
        print("Initializing RAG and Quiz systems...")
        rag_system = RAGSystem(vector_store)
        quiz_generator = QuizGenerator(vector_store)
        
        print(f"✓ Upload complete!")
        print(f"{'='*60}\n")
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Ask a question using RAG"""
    if rag_system is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents uploaded yet. Please upload documents first."
        )
    
    try:
        print(f"\n[Q&A REQUEST] {request.question}")
        result = rag_system.ask_question(request.question, k=request.k)
        print(f"[Q&A RESPONSE] Generated answer with {len(result['sources'])} sources")
        return result
    except Exception as e:
        print(f"✗ Q&A failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.post("/summarize")
def summarize(request: SummarizeRequest):
    """Summarize content from uploaded documents"""
    if rag_system is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents uploaded yet. Please upload documents first."
        )
    
    try:
        print(f"\n[SUMMARY REQUEST] Type: {request.summary_type}, Topic: {request.topic}")
        result = rag_system.summarize(
            query=request.topic,
            summary_type=request.summary_type,
            k=request.k
        )
        print(f"[SUMMARY RESPONSE] Generated from {len(result['sources'])} sources")
        return result
    except Exception as e:
        print(f"✗ Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/definitions")
def get_definitions(topic: str = "definitions terms concepts"):
    """Extract key definitions and terms from uploaded materials"""
    if rag_system is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents uploaded yet. Please upload documents first."
        )
    
    try:
        print(f"\n[DEFINITIONS REQUEST] Topic: {topic}")
        result = rag_system.extract_definitions(query=topic)
        print(f"[DEFINITIONS RESPONSE] Extracted from {len(result['sources'])} sources")
        return result
    except Exception as e:
        print(f"✗ Definition extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Definition extraction failed: {str(e)}")


@app.post("/quiz/generate")
def generate_quiz(request: QuizRequest):
    """Generate a quiz from uploaded materials"""
    if quiz_generator is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents uploaded yet. Please upload documents first."
        )
    
    try:
        print(f"\n[QUIZ REQUEST] Topic: {request.topic}, Questions: {request.num_questions}, Difficulty: {request.difficulty}")
        
        quiz = quiz_generator.generate_quiz(
            topic=request.topic,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )
        
        if "error" in quiz:
            raise HTTPException(status_code=500, detail=quiz["error"])
        
        print(f"[QUIZ RESPONSE] Generated {len(quiz.get('questions', []))} questions")
        return quiz
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Quiz generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")


@app.post("/quiz/grade")
def grade_quiz(request: GradeQuizRequest):
    """Grade a quiz submission"""
    if quiz_generator is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents uploaded yet. Please upload documents first."
        )
    
    try:
        print(f"\n[QUIZ GRADING] Grading {len(request.questions)} questions")
        results = quiz_generator.grade_quiz(request.questions, request.user_answers)
        print(f"[QUIZ GRADING] Score: {results['score']}%")
        return results
    except Exception as e:
        print(f"✗ Quiz grading failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quiz grading failed: {str(e)}")


@app.get("/documents")
def list_documents():
    """List all uploaded documents"""
    try:
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        return {
            "documents": files,
            "count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.delete("/reset")
def reset_system():
    """Reset the system (clear all data)"""
    global vector_store, rag_system, quiz_generator
    
    try:
        # Clear uploads
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Clear vector store
        vector_store_path = os.path.join(parent_dir, "data", "vector_store", VECTOR_STORE_NAME)
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
        
        # Reset globals
        vector_store = None
        rag_system = None
        quiz_generator = None
        
        return {
            "message": "System reset successfully",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# Run the API
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("AI STUDY ASSISTANT API")
    print("="*60)
    print("Powered by: Ollama")
    print("Starting server on: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")