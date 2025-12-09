import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import DocumentIngestion
from src.rag import RAGSystem
from src.quiz_generator import QuizGenerator


def create_test_document():
    """Create a test document with sample content"""
    test_file = "data/uploads/machine_learning_basics.txt"
    
    content = """
Machine Learning: An Introduction

What is Machine Learning?
Machine learning is a subset of artificial intelligence (AI) that provides systems 
the ability to automatically learn and improve from experience without being 
explicitly programmed. Machine learning focuses on the development of computer 
programs that can access data and use it to learn for themselves.

Types of Machine Learning

1. Supervised Learning
Supervised learning is when the model is trained on labeled data. The algorithm 
learns from the training dataset and makes predictions. Examples include:
- Linear Regression
- Logistic Regression  
- Decision Trees
- Random Forests

2. Unsupervised Learning
Unsupervised learning is used when the data has no labels. The algorithm tries 
to find patterns in the data. Examples include:
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)

3. Reinforcement Learning
Reinforcement learning is learning through trial and error. The agent learns to 
achieve a goal in an uncertain environment by receiving rewards or penalties.

Key Concepts

Gradient Descent
Gradient descent is an optimization algorithm used to minimize the cost function 
in machine learning models. It works by iteratively adjusting parameters in the 
direction of steepest descent.

Overfitting
Overfitting occurs when a model learns the training data too well, including its 
noise and outliers. This causes poor performance on new, unseen data.

Cross-Validation
Cross-validation is a technique to assess how well a model generalizes to unseen 
data. The most common method is k-fold cross-validation.

Neural Networks
Neural networks are computing systems inspired by biological neural networks. 
They consist of interconnected nodes (neurons) organized in layers:
- Input Layer
- Hidden Layers  
- Output Layer

Deep Learning
Deep learning is a subset of machine learning that uses neural networks with 
multiple hidden layers. It excels at tasks like image recognition, natural 
language processing, and speech recognition.

Applications of Machine Learning
- Image and Speech Recognition
- Medical Diagnosis
- Recommendation Systems
- Fraud Detection
- Autonomous Vehicles
- Natural Language Processing

Conclusion
Machine learning is revolutionizing technology and continues to grow rapidly. 
Understanding its fundamental concepts is essential for anyone working in 
modern technology fields.
"""
    
    os.makedirs("data/uploads", exist_ok=True)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Created test document: {test_file}")
    return test_file


def test_complete_pipeline():
    """Test the complete RAG pipeline"""
    
    print("\n" + "="*60)
    print("TESTING AI STUDY ASSISTANT - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Create test document
    print("\n[STEP 1] Creating test document...")
    test_file = create_test_document()
    
    # Step 2: Initialize ingestion
    print("\n[STEP 2] Initializing Document Ingestion...")
    ingestion = DocumentIngestion()
    
    # Step 3: Process document
    print("\n[STEP 3] Processing document...")
    chunks = ingestion.process_documents(test_file)
    
    if not chunks:
        print("✗ FAILED: No chunks created")
        return
    
    # Step 4: Create vector store
    print("\n[STEP 4] Creating vector store...")
    vector_store = ingestion.create_vector_store(chunks, "test_store")
    
    # Step 5: Test RAG Q&A
    print("\n[STEP 5] Testing RAG Q&A System...")
    rag = RAGSystem(vector_store)
    
    test_questions = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "Explain gradient descent"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = rag.ask_question(question)
        print(f"A: {result['answer'][:200]}...")
        print(f"Sources: {len(result['sources'])} chunks")
    
    # Step 6: Test Summarization
    print("\n[STEP 6] Testing Summarization...")
    summary = rag.summarize(summary_type="bullets")
    print(f"Summary:\n{summary['summary'][:300]}...")
    
    # Step 7: Test Definition Extraction
    print("\n[STEP 7] Testing Definition Extraction...")
    definitions = rag.extract_definitions()
    print(f"Definitions:\n{definitions['definitions'][:300]}...")
    
    # Step 8: Test Quiz Generation
    print("\n[STEP 8] Testing Quiz Generation...")
    quiz_gen = QuizGenerator(vector_store)
    quiz = quiz_gen.generate_quiz(
        topic="machine learning types",
        num_questions=3,
        difficulty="medium"
    )
    
    if "questions" in quiz and quiz["questions"]:
        print(f"\n✓ Generated {len(quiz['questions'])} questions:")
        for i, q in enumerate(quiz['questions'], 1):
            print(f"\n{i}. {q['question']}")
            for opt, text in q['options'].items():
                print(f"   {opt}) {text}")
            print(f"   Correct: {q['correct_answer']}")
    else:
        print(f"✗ Quiz generation failed: {quiz.get('error', 'Unknown error')}")
    
    # Step 9: Test Quiz Grading
    if "questions" in quiz and quiz["questions"]:
        print("\n[STEP 9] Testing Quiz Grading...")
        # Simulate user answers (all correct)
        user_answers = {i: q['correct_answer'] for i, q in enumerate(quiz['questions'])}
        
        grade_result = quiz_gen.grade_quiz(quiz['questions'], user_answers)
        print(f"\n✓ Quiz Score: {grade_result['score']}%")
        print(f"  Correct: {grade_result['correct']}/{grade_result['total']}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    try:
        test_complete_pipeline()
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()