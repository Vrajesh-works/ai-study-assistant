"""Prompt templates optimized for Ollama models"""

QA_PROMPT_TEMPLATE = """You are a helpful tutoring assistant. Answer the student's question using ONLY the context provided below.

Rules:
1. If the answer is in the context, provide a clear explanation
2. If the answer is NOT in the context, say "I don't have enough information from your materials to answer this."
3. Keep answers concise but complete
4. Cite the source when possible

Context:
{context}

Question: {question}

Answer:"""

SUMMARIZATION_PROMPT_TEMPLATE = """Summarize the following study material content.

Content:
{context}

Create a {summary_type} summary:
- short: 2-3 sentences
- bullets: 5-7 bullet points
- detailed: comprehensive paragraph
- eli15: explain like I'm 15 years old

Summary:"""

QUIZ_GENERATION_PROMPT = """Generate {num_questions} multiple-choice questions from this content. Difficulty: {difficulty}

Content:
{context}

IMPORTANT: Return ONLY valid JSON with no extra text. Use this exact format:

{{
  "questions": [
    {{
      "question": "What is X?",
      "options": {{
        "A": "option 1",
        "B": "option 2", 
        "C": "option 3",
        "D": "option 4"
      }},
      "correct_answer": "A",
      "explanation": "Brief explanation"
    }}
  ]
}}

JSON:"""

DEFINITION_EXTRACTION_PROMPT = """Extract all key definitions and terms from this content.

Content:
{context}

Format each as:
Term: [term name]
Definition: [clear definition]

List:"""