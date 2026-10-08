"""Unit tests for quiz generation helpers and grading (no LLM required)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quiz_generator import QuizGenerator

SAMPLE_QUESTIONS = [
    {
        "question": "What is machine learning?",
        "options": {"A": "A type of database", "B": "AI that learns from data",
                    "C": "A web framework", "D": "An OS"},
        "correct_answer": "B",
        "explanation": "ML is a subset of AI focused on learning from data.",
    },
    {
        "question": "What does PCA stand for?",
        "options": {"A": "Principal Component Analysis", "B": "Personal Computer Aid",
                    "C": "Primary Component Algorithm", "D": "None"},
        "correct_answer": "A",
        "explanation": "PCA is a dimensionality reduction technique.",
    },
]


def test_extract_json_from_markdown_block():
    text = 'Here you go:\n```json\n{"questions": [{"q": 1}]}\n```\nDone.'
    result = QuizGenerator._extract_json(text)
    assert result == '{"questions": [{"q": 1}]}'


def test_extract_json_from_plain_text():
    text = 'Some preamble {"questions": []} some trailing text'
    result = QuizGenerator._extract_json(text)
    assert result == '{"questions": []}'


def test_extract_json_no_json_returns_stripped():
    assert QuizGenerator._extract_json("   hello   ") == "hello"


def test_grade_quiz_all_correct():
    answers = {0: "B", 1: "A"}
    result = QuizGenerator.grade_quiz(SAMPLE_QUESTIONS, answers)
    assert result["score"] == 100.0
    assert result["correct"] == 2
    assert result["total"] == 2
    assert all(r["is_correct"] for r in result["results"])


def test_grade_quiz_partial():
    answers = {0: "B", 1: "C"}
    result = QuizGenerator.grade_quiz(SAMPLE_QUESTIONS, answers)
    assert result["score"] == 50.0
    assert result["correct"] == 1
    assert result["results"][0]["is_correct"] is True
    assert result["results"][1]["is_correct"] is False


def test_grade_quiz_case_insensitive():
    answers = {0: "b", 1: "a"}
    result = QuizGenerator.grade_quiz(SAMPLE_QUESTIONS, answers)
    assert result["score"] == 100.0


def test_grade_quiz_missing_answer_marked_wrong():
    answers = {0: "B"}
    result = QuizGenerator.grade_quiz(SAMPLE_QUESTIONS, answers)
    assert result["score"] == 50.0
    assert result["results"][1]["user_answer"] == "Not answered"
    assert result["results"][1]["is_correct"] is False


def test_grade_quiz_empty_questions():
    result = QuizGenerator.grade_quiz([], {})
    assert result["score"] == 0
    assert result["total"] == 0
