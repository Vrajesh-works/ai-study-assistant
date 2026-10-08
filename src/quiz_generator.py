"""Quiz generation and grading backed by Ollama."""

import json
import logging
import re

from langchain_community.vectorstores import Chroma

from src.llm import get_llm
from src.prompts import QUIZ_GENERATION_PROMPT

logger = logging.getLogger(__name__)

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


class QuizGenerator:
    """Generate multiple-choice quizzes from study materials."""

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

        logger.info("Initializing quiz generator")
        self.llm = get_llm(temperature=0.7, max_tokens=2048)
        logger.info("Quiz generator ready")

    def generate_quiz(
        self,
        topic: str,
        num_questions: int = 10,
        difficulty: str = "medium",
        k: int = 15,
    ) -> dict:
        """Generate a multiple-choice quiz on a topic."""
        difficulty = difficulty.lower()
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. "
                f"Choose from: {sorted(VALID_DIFFICULTIES)}"
            )
        num_questions = max(1, min(10, num_questions))

        logger.info(
            "Generating quiz: topic=%s questions=%d difficulty=%s",
            topic,
            num_questions,
            difficulty,
        )
        relevant_docs = self.vector_store.similarity_search(topic, k=k)

        if not relevant_docs:
            return {
                "error": "No relevant content found for quiz generation",
                "questions": [],
            }
        logger.info("Found %d relevant chunks", len(relevant_docs))

        context = "\n\n".join(doc.page_content for doc in relevant_docs[:8])
        prompt = QUIZ_GENERATION_PROMPT.format(
            num_questions=num_questions,
            context=context[:3000],
            difficulty=difficulty,
        )

        try:
            quiz_text = self.llm.invoke(prompt)
            quiz_data = json.loads(self._extract_json(quiz_text))

            if "questions" not in quiz_data or not isinstance(
                quiz_data["questions"], list
            ):
                raise ValueError("Invalid quiz structure")

            quiz_data["metadata"] = {
                "topic": topic,
                "difficulty": difficulty,
                "num_questions": len(quiz_data.get("questions", [])),
                "sources": list(
                    {
                        doc.metadata.get("source", "unknown")
                        for doc in relevant_docs[:5]
                    }
                ),
            }
            logger.info(
                "Generated %d questions", len(quiz_data["questions"])
            )
            return quiz_data

        except json.JSONDecodeError as e:
            logger.warning("Quiz JSON parsing failed: %s", e)
            return {
                "error": f"Failed to parse quiz JSON: {e}",
                "questions": [],
            }
        except Exception:
            logger.exception("Quiz generation failed")
            return {"error": "Quiz generation failed", "questions": []}

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON object from a model response."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            for part in text.split("```"):
                if "{" in part and "}" in part:
                    text = part
                    break

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        return text.strip()

    @staticmethod
    def grade_quiz(questions: list[dict], user_answers: dict) -> dict:
        """Grade a quiz submission. Pure logic, no LLM call."""
        logger.info("Grading quiz with %d questions", len(questions))
        results = []
        correct_count = 0

        for idx, question in enumerate(questions):
            raw_answer = user_answers.get(idx, user_answers.get(str(idx), ""))
            user_answer = str(raw_answer).upper()
            correct_answer = str(question.get("correct_answer", "")).upper()
            is_correct = bool(user_answer) and user_answer == correct_answer

            if is_correct:
                correct_count += 1

            results.append(
                {
                    "question_number": idx + 1,
                    "question": question.get("question"),
                    "user_answer": user_answer if user_answer else "Not answered",
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "explanation": question.get("explanation", ""),
                }
            )

        score = (correct_count / len(questions)) * 100 if questions else 0
        logger.info("Quiz score: %.1f%% (%d/%d)", score, correct_count, len(questions))

        return {
            "score": round(score, 1),
            "correct": correct_count,
            "total": len(questions),
            "percentage": f"{score:.1f}%",
            "results": results,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Quiz Generator Module - Ready!")
    print("\nTo test:")
    print("1. Load a vector store")
    print("2. Create quiz generator:")
    print("\n  from src.quiz_generator import QuizGenerator")
    print("  quiz_gen = QuizGenerator(vector_store)")
    print("  quiz = quiz_gen.generate_quiz('machine learning', num_questions=3)")
    print("  print(quiz)")
