import json
import re
from typing import List, Dict
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from src.prompts import QUIZ_GENERATION_PROMPT


class QuizGenerator:
    """Generate quizzes from study materials using Ollama"""
    
    def __init__(self, vector_store: Chroma, model_name: str = "llama3.2"):
        self.vector_store = vector_store
        
        print(f"Initializing Quiz Generator with {model_name}")
        
        self.llm = Ollama(
            model=model_name,
            temperature=0.7,  # Slightly higher for creative question generation
            base_url="http://localhost:11434",
            num_predict=2048,  # Allow longer responses for multiple questions
        )
        
        print("✓ Quiz Generator ready")
    
    def generate_quiz(
        self, 
        topic: str, 
        num_questions: int = 10, 
        difficulty: str = "medium",
        k: int = 15
    ) -> Dict:
        """
        Generate a quiz on a specific topic
        
        Args:
            topic: Topic or query for quiz generation
            num_questions: Number of questions (1-10 recommended)
            difficulty: "easy", "medium", or "hard"
            k: Number of document chunks to retrieve
        """
        print(f"\n{'='*50}")
        print(f"Quiz Generation")
        print(f"{'='*50}")
        print(f"Topic: {topic}")
        print(f"Questions: {num_questions}")
        print(f"Difficulty: {difficulty}")
        
        # Retrieve relevant content
        print("Searching for relevant content...")
        relevant_docs = self.vector_store.similarity_search(topic, k=k)
        
        if not relevant_docs:
            return {
                "error": "No relevant content found for quiz generation",
                "questions": []
            }
        
        print(f"✓ Found {len(relevant_docs)} relevant chunks")
        
        # Prepare context (limit to avoid token limits)
        context = "\n\n".join([doc.page_content for doc in relevant_docs[:8]])
        context = context[:3000]  # Limit context size
        
        # Create prompt
        prompt = QUIZ_GENERATION_PROMPT.format(
            num_questions=num_questions,
            context=context,
            difficulty=difficulty
        )
        
        # Generate quiz
        print("Generating quiz questions...")
        try:
            quiz_text = self.llm.invoke(prompt)
            
            print("✓ Response received")
            print("Parsing JSON...")
            
            # Clean up response - extract JSON
            quiz_text = self._extract_json(quiz_text)
            
            # Parse JSON
            quiz_data = json.loads(quiz_text)
            
            # Validate structure
            if "questions" not in quiz_data or not isinstance(quiz_data["questions"], list):
                raise ValueError("Invalid quiz structure")
            
            # Add metadata
            quiz_data["metadata"] = {
                "topic": topic,
                "difficulty": difficulty,
                "num_questions": len(quiz_data.get("questions", [])),
                "sources": list(set([doc.metadata.get('source', 'unknown') for doc in relevant_docs[:5]]))
            }
            
            print(f"✓ Successfully generated {len(quiz_data['questions'])} questions")
            return quiz_data
            
        except json.JSONDecodeError as e:
            print(f"✗ JSON parsing failed: {str(e)}")
            return {
                "error": f"Failed to parse quiz JSON: {str(e)}",
                "raw_response": quiz_text if 'quiz_text' in locals() else "No response",
                "questions": []
            }
        except Exception as e:
            print(f"✗ Quiz generation failed: {str(e)}")
            return {
                "error": f"Quiz generation failed: {str(e)}",
                "questions": []
            }
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from model response"""
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            # Try to extract any code block
            parts = text.split("```")
            for part in parts:
                if "{" in part and "}" in part:
                    text = part
                    break
        
        # Find JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        
        return text.strip()
    
    def grade_quiz(self, questions: List[Dict], user_answers: Dict[int, str]) -> Dict:
        """
        Grade a quiz submission
        
        Args:
            questions: List of question dictionaries
            user_answers: Dict mapping question index to answer (A/B/C/D)
        """
        print(f"\n{'='*50}")
        print("Grading Quiz")
        print(f"{'='*50}")
        
        results = []
        correct_count = 0
        
        for idx, question in enumerate(questions):
            user_answer = user_answers.get(idx, "").upper()
            correct_answer = question.get("correct_answer", "").upper()
            is_correct = user_answer == correct_answer
            
            if is_correct:
                correct_count += 1
            
            results.append({
                "question_number": idx + 1,
                "question": question.get("question"),
                "user_answer": user_answer if user_answer else "Not answered",
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "explanation": question.get("explanation", "")
            })
        
        score = (correct_count / len(questions)) * 100 if questions else 0
        
        print(f"✓ Score: {score:.1f}% ({correct_count}/{len(questions)})")
        
        return {
            "score": round(score, 1),
            "correct": correct_count,
            "total": len(questions),
            "percentage": f"{score:.1f}%",
            "results": results
        }


# Test the module
if __name__ == "__main__":
    print("Quiz Generator Module - Ready!")
    print("\nTo test:")
    print("1. Load a vector store")
    print("2. Create quiz generator:")
    print("\n  from src.quiz_generator import QuizGenerator")
    print("  quiz_gen = QuizGenerator(vector_store)")
    print("  quiz = quiz_gen.generate_quiz('machine learning', num_questions=3)")
    print("  print(quiz)")