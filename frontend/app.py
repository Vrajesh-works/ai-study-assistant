import streamlit as st
import requests
import json
from typing import Dict, List
import time

# API Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .quiz-question {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
    }
    .correct-answer {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .wrong-answer {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.json()
    except:
        return None


def upload_document(file):
    """Upload document to API"""
    files = {"file": (file.name, file, file.type)}
    response = requests.post(f"{API_URL}/upload", files=files)
    return response.json()


def ask_question(question: str, k: int = 5):
    """Ask a question"""
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question, "k": k}
    )
    return response.json()


def get_summary(topic: str = None, summary_type: str = "bullets", k: int = 10):
    """Get summary"""
    response = requests.post(
        f"{API_URL}/summarize",
        json={"topic": topic, "summary_type": summary_type, "k": k}
    )
    return response.json()


def get_definitions(topic: str = "definitions"):
    """Get definitions"""
    response = requests.post(
        f"{API_URL}/definitions",
        params={"topic": topic}
    )
    return response.json()


def generate_quiz(topic: str, num_questions: int, difficulty: str):
    """Generate quiz"""
    response = requests.post(
        f"{API_URL}/quiz/generate",
        json={
            "topic": topic,
            "num_questions": num_questions,
            "difficulty": difficulty
        }
    )
    return response.json()


def grade_quiz(questions: List[Dict], user_answers: Dict[int, str]):
    """Grade quiz"""
    response = requests.post(
        f"{API_URL}/quiz/grade",
        json={
            "questions": questions,
            "user_answers": user_answers
        }
    )
    return response.json()


def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 AI Study Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by Ollama - 100% FREE & Local</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Check API status
        api_status = check_api_status()
        
        if api_status:
            if api_status.get('documents_loaded'):
                st.success("✅ System Ready")
                st.info(f"🤖 Using Ollama (Local)")
            else:
                st.warning("⚠️ No documents loaded")
        else:
            st.error("❌ API not reachable")
            st.error("Please start the API server:")
            st.code("python api/main.py", language="bash")
            st.stop()
        
        st.divider()
        
        # File upload
        st.subheader("Upload Study Materials")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt"],
            help="Upload PDFs or text files containing your study materials"
        )
        
        if uploaded_file:
            if st.button("📤 Process Document", use_container_width=True):
                with st.spinner("Processing document... This may take a minute..."):
                    try:
                        result = upload_document(uploaded_file)
                        st.success(f"✅ {result['message']}")
                        st.info(f"📊 Created {result['chunks_created']} chunks")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Display uploaded documents
        try:
            docs_response = requests.get(f"{API_URL}/documents")
            docs_data = docs_response.json()
            
            if docs_data['documents']:
                st.subheader(f"📚 Uploaded Files ({docs_data['count']})")
                for doc in docs_data['documents']:
                    st.text(f"• {doc}")
        except:
            pass
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset System", use_container_width=True, type="secondary"):
            if st.confirm("⚠️ This will delete all data. Continue?"):
                try:
                    requests.delete(f"{API_URL}/reset")
                    st.success("System reset!")
                    time.sleep(1)
                    st.rerun()
                except:
                    st.error("Reset failed")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Questions", "📝 Summarize", "🎯 Quiz Me", "📚 Definitions"])
    
    # Tab 1: Q&A
    with tab1:
        st.header("Ask Questions About Your Study Materials")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What is machine learning? Explain gradient descent.",
                label_visibility="collapsed"
            )
        
        with col2:
            k_value = st.number_input("Sources", min_value=1, max_value=10, value=5, help="Number of relevant chunks to retrieve")
        
        if st.button("🔍 Get Answer", use_container_width=True, type="primary"):
            if question:
                with st.spinner("Searching knowledge base and generating answer..."):
                    try:
                        result = ask_question(question, k=k_value)
                        
                        st.markdown("### 💡 Answer")
                        st.markdown(result['answer'])
                        
                        if result['sources']:
                            with st.expander(f"📖 View {len(result['sources'])} Sources"):
                                for i, source in enumerate(result['sources'], 1):
                                    st.markdown(f"**Source {i}: {source['source']} (Page {source['page']})**")
                                    st.text(source['excerpt'])
                                    if i < len(result['sources']):
                                        st.divider()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question")
        
        # Example questions
        st.divider()
        st.subheader("💡 Example Questions")
        example_questions = [
            "What are the main concepts in this material?",
            "Explain [topic] with an example",
            "What is the difference between [concept A] and [concept B]?",
            "Summarize the key points about [topic]"
        ]
        
        cols = st.columns(2)
        for idx, eq in enumerate(example_questions):
            with cols[idx % 2]:
                if st.button(eq, key=f"example_{idx}", use_container_width=True):
                    st.session_state['question_input'] = eq
    
    # Tab 2: Summarization
    with tab2:
        st.header("Generate Summaries")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            topic = st.text_input(
                "Topic (optional):",
                placeholder="e.g., Chapter 3, Machine Learning, or leave empty for general summary",
                help="Specify a topic to focus the summary, or leave empty for a general overview"
            )
        
        with col2:
            summary_type = st.selectbox(
                "Summary type:",
                ["bullets", "short", "detailed", "eli15"],
                help="bullets: 5-7 bullet points\nshort: 2-3 sentences\ndetailed: comprehensive\neli15: simple explanation"
            )
        
        if st.button("📝 Generate Summary", use_container_width=True, type="primary"):
            with st.spinner("Generating summary..."):
                try:
                    result = get_summary(topic if topic else None, summary_type)
                    
                    st.markdown("### 📄 Summary")
                    st.markdown(result['summary'])
                    
                    if result['sources']:
                        st.info(f"📚 Based on: {', '.join(result['sources'])}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Tab 3: Quiz
    with tab3:
        st.header("Test Your Knowledge")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quiz_topic = st.text_input(
                "Quiz topic:",
                placeholder="e.g., Chapter 2, Neural Networks",
                help="Topic to generate questions about"
            )
        
        with col2:
            num_questions = st.number_input(
                "Number of questions:",
                min_value=1,
                max_value=10,
                value=5,
                help="Recommended: 3-5 questions for best results"
            )
        
        with col3:
            difficulty = st.selectbox(
                "Difficulty:",
                ["easy", "medium", "hard"],
                index=1
            )
        
        if st.button("🎯 Generate Quiz", use_container_width=True, type="primary"):
            if quiz_topic:
                with st.spinner(f"Generating {num_questions} {difficulty} questions... This may take a minute..."):
                    try:
                        quiz_data = generate_quiz(quiz_topic, num_questions, difficulty)
                        
                        if 'questions' in quiz_data and quiz_data['questions']:
                            st.session_state['current_quiz'] = quiz_data
                            st.success(f"✅ Generated {len(quiz_data['questions'])} questions!")
                            st.rerun()
                        else:
                            st.error(f"❌ Quiz generation failed: {quiz_data.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a quiz topic")
        
        # Display quiz if generated
        if 'current_quiz' in st.session_state:
            quiz_data = st.session_state['current_quiz']
            
            st.divider()
            st.subheader("📝 Answer the Questions")
            
            # Show quiz metadata
            if 'metadata' in quiz_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Topic", quiz_data['metadata']['topic'])
                with col2:
                    st.metric("Questions", quiz_data['metadata']['num_questions'])
                with col3:
                    st.metric("Difficulty", quiz_data['metadata']['difficulty'])
            
            st.divider()
            
            user_answers = {}
            
            for i, q in enumerate(quiz_data['questions']):
                st.markdown(f"**Question {i+1} of {len(quiz_data['questions'])}**")
                st.markdown(f"### {q['question']}")
                
                options_list = [f"{key}: {value}" for key, value in q['options'].items()]
                
                answer = st.radio(
                    "Select your answer:",
                    options=list(q['options'].keys()),
                    format_func=lambda x: f"{x}. {q['options'][x]}",
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                
                user_answers[i] = answer
                st.divider()
            
            if st.button("✅ Submit Quiz", use_container_width=True, type="primary"):
                with st.spinner("Grading your quiz..."):
                    try:
                        results = grade_quiz(quiz_data['questions'], user_answers)
                        
                        # Show score
                        score = results['score']
                        if score >= 80:
                            st.balloons()
                            st.success(f"🎉 Excellent! Score: {score}% ({results['correct']}/{results['total']})")
                        elif score >= 60:
                            st.info(f"👍 Good job! Score: {score}% ({results['correct']}/{results['total']})")
                        else:
                            st.warning(f"📚 Keep studying! Score: {score}% ({results['correct']}/{results['total']})")
                        
                        # Detailed results
                        with st.expander("📊 View Detailed Results", expanded=True):
                            for result in results['results']:
                                is_correct = result['is_correct']
                                icon = "✅" if is_correct else "❌"
                                
                                st.markdown(f"{icon} **Question {result['question_number']}:** {result['question']}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Your answer:** {result['user_answer']}")
                                with col2:
                                    st.write(f"**Correct answer:** {result['correct_answer']}")
                                
                                if not is_correct:
                                    st.info(f"💡 **Explanation:** {result['explanation']}")
                                
                                st.divider()
                        
                        # Clear quiz after submission
                        if st.button("🔄 Take Another Quiz"):
                            del st.session_state['current_quiz']
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Grading failed: {str(e)}")
    
    # Tab 4: Definitions
    with tab4:
        st.header("Extract Key Definitions")
        
        def_topic = st.text_input(
            "Topic (optional):",
            placeholder="e.g., neural networks, or leave empty for all definitions",
            help="Specify a topic to extract related definitions"
        )
        
        if st.button("📚 Extract Definitions", use_container_width=True, type="primary"):
            with st.spinner("Extracting definitions from your materials..."):
                try:
                    result = get_definitions(def_topic if def_topic else "definitions terms concepts")
                    
                    st.markdown("### 📖 Definitions & Key Terms")
                    st.markdown(result['definitions'])
                    
                    if result['sources']:
                        st.info(f"📚 Extracted from: {', '.join(result['sources'])}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>AI Study Assistant</strong> - Powered by Ollama (100% Free & Open Source)</p>
        <p>Your data stays on your computer. No cloud, no tracking, no limits.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()