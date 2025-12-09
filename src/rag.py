from typing import List, Dict, Optional
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from src.prompts import (
    QA_PROMPT_TEMPLATE, 
    SUMMARIZATION_PROMPT_TEMPLATE, 
    DEFINITION_EXTRACTION_PROMPT
)


class RAGSystem:
    """RAG-based Q&A system using Ollama"""
    
    def __init__(self, vector_store: Chroma, model_name: str = "llama3.2", temperature: float = 0.3):
        self.vector_store = vector_store
        
        print(f"Initializing RAG with Ollama model: {model_name}")
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
            num_predict=512,  # Max tokens to generate
        )
        
        # Setup retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        print("✓ RAG System ready")
    
    def ask_question(self, question: str, k: int = 5) -> Dict:
        """Answer a question using RAG"""
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"{'='*50}")
        
        # Retrieve relevant documents
        print("Searching for relevant content...")
        relevant_docs = self.vector_store.similarity_search(question, k=k)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information in your study materials.",
                "sources": []
            }
        
        print(f"✓ Found {len(relevant_docs)} relevant chunks")
        
        # Prepare context
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])
        
        # Create prompt
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Get answer from Ollama
        print("Generating answer...")
        answer = self.llm.invoke(prompt)
        
        print("✓ Answer generated")
        
        # Prepare sources
        sources = [
            {
                "source": doc.metadata.get('source', 'unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "excerpt": doc.page_content[:200] + "..."
            }
            for doc in relevant_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def summarize(self, query: str = None, summary_type: str = "bullets", k: int = 10) -> Dict:
        """Summarize content from the knowledge base"""
        print(f"\n{'='*50}")
        print(f"Summarization Request: {summary_type}")
        print(f"{'='*50}")
        
        if query:
            print(f"Topic: {query}")
            relevant_docs = self.vector_store.similarity_search(query, k=k)
        else:
            print("Generating general summary")
            # Get diverse chunks for general summary
            relevant_docs = self.vector_store.similarity_search("overview main concepts key topics", k=k)
        
        if not relevant_docs:
            return {
                "summary": "No content found to summarize.",
                "sources": []
            }
        
        print(f"✓ Found {len(relevant_docs)} relevant chunks")
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(
            context=context[:4000],  # Limit context size
            summary_type=summary_type
        )
        
        # Generate summary
        print("Generating summary...")
        summary = self.llm.invoke(prompt)
        
        print("✓ Summary generated")
        
        # Prepare sources
        sources = list(set([doc.metadata.get('source', 'unknown') for doc in relevant_docs]))
        
        return {
            "summary": summary,
            "sources": sources
        }
    
    def extract_definitions(self, query: str = "definitions terms concepts", k: int = 10) -> Dict:
        """Extract key definitions from content"""
        print(f"\n{'='*50}")
        print("Extracting Definitions")
        print(f"{'='*50}")
        
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        
        if not relevant_docs:
            return {
                "definitions": "No definitions found.",
                "sources": []
            }
        
        print(f"✓ Found {len(relevant_docs)} relevant chunks")
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = DEFINITION_EXTRACTION_PROMPT.format(context=context[:4000])
        
        print("Extracting definitions...")
        definitions = self.llm.invoke(prompt)
        
        print("✓ Definitions extracted")
        
        sources = list(set([doc.metadata.get('source', 'unknown') for doc in relevant_docs]))
        
        return {
            "definitions": definitions,
            "sources": sources
        }


# Test the module
if __name__ == "__main__":
    print("RAG System Module - Ready!")
    print("\nTo test:")
    print("1. First create a vector store using ingestion.py")
    print("2. Then run:")
    print("\n  from src.ingestion import DocumentIngestion")
    print("  from src.rag import RAGSystem")
    print("  ingestion = DocumentIngestion()")
    print("  vector_store = ingestion.load_vector_store('test_store')")
    print("  rag = RAGSystem(vector_store)")
    print("  result = rag.ask_question('What is machine learning?')")
    print("  print(result['answer'])")