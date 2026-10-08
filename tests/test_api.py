"""API contract tests (no Ollama required).

Validates request schemas and the not-ready guard. Endpoints that need a
loaded vector store must return 400 before any document is uploaded.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_status():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["documents_loaded"] is False
    assert "endpoints" in body


def test_health_endpoint_shape():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "ollama_reachable" in body
    assert body["documents_loaded"] is False


def test_ask_requires_documents():
    response = client.post("/ask", json={"question": "What is ML?"})
    assert response.status_code == 400


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_summarize_requires_documents():
    response = client.post("/summarize", json={"summary_type": "bullets"})
    assert response.status_code == 400


def test_summarize_rejects_bad_type():
    response = client.post("/summarize", json={"summary_type": "tweet"})
    assert response.status_code == 422


def test_definitions_requires_documents():
    response = client.post("/definitions", json={"topic": "neural networks"})
    assert response.status_code == 400


def test_quiz_generate_validates_input():
    # num_questions out of range
    response = client.post(
        "/quiz/generate",
        json={"topic": "ML", "num_questions": 99, "difficulty": "medium"},
    )
    assert response.status_code == 422

    # invalid difficulty
    response = client.post(
        "/quiz/generate",
        json={"topic": "ML", "num_questions": 3, "difficulty": "extreme"},
    )
    assert response.status_code == 422


def test_quiz_generate_requires_documents():
    response = client.post(
        "/quiz/generate",
        json={"topic": "ML", "num_questions": 3, "difficulty": "easy"},
    )
    assert response.status_code == 400


def test_quiz_grade_requires_documents():
    response = client.post("/quiz/grade", json={"questions": [], "user_answers": {}})
    assert response.status_code == 400


def test_upload_rejects_unsupported_type():
    response = client.post(
        "/upload",
        files={"file": ("notes.docx", b"fake", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_documents_lists_uploads():
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert "documents" in body and "count" in body
