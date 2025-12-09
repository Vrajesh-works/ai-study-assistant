import os
from typing import List, Dict
from pathlib import Path
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import json


class DocumentIngestion:
    """Handles document upload, processing, and indexing with Ollama"""
    
    def __init__(self, vector_store_path: str = "data/vector_store"):
        self.vector_store_path = vector_store_path
        
        # Initialize Ollama embeddings
        print("Initializing Ollama embeddings...")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page numbers"""
        documents = []
        
        try:
            print(f"Extracting text from {pdf_path}...")
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                filename = Path(pdf_path).name
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    print(f"  Processing page {page_num}/{total_pages}...", end='\r')
                    text = page.extract_text()
                    
                    if text.strip():
                        documents.append({
                            'content': text,
                            'metadata': {
                                'source': filename,
                                'page': page_num,
                                'type': 'pdf'
                            }
                        })
                
                print(f"\n✓ Extracted {len(documents)} pages from {filename}")
                        
        except Exception as e:
            print(f"✗ Error extracting PDF {pdf_path}: {str(e)}")
            
        return documents
    
    def extract_text_from_txt(self, txt_path: str) -> List[Dict]:
        """Extract text from plain text file"""
        try:
            print(f"Reading text file {txt_path}...")
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
                filename = Path(txt_path).name
                
                print(f"✓ Read {len(text)} characters from {filename}")
                return [{
                    'content': text,
                    'metadata': {
                        'source': filename,
                        'type': 'txt'
                    }
                }]
        except Exception as e:
            print(f"✗ Error reading text file {txt_path}: {str(e)}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove null characters
        text = text.replace('\x00', '')
        # Remove very short lines that are likely artifacts
        return text.strip()
    
    def process_documents(self, file_path: str) -> List[Document]:
        """Process a document into chunks"""
        print(f"\n{'='*50}")
        print(f"Processing: {file_path}")
        print(f"{'='*50}")
        
        file_extension = Path(file_path).suffix.lower()
        
        # Extract text based on file type
        if file_extension == '.pdf':
            raw_docs = self.extract_text_from_pdf(file_path)
        elif file_extension == '.txt':
            raw_docs = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if not raw_docs:
            raise ValueError("No content extracted from document")
        
        # Clean and create LangChain documents
        print("Cleaning text...")
        documents = []
        for doc in raw_docs:
            cleaned_text = self.clean_text(doc['content'])
            if cleaned_text and len(cleaned_text) > 50:  # Skip very short chunks
                documents.append(Document(
                    page_content=cleaned_text,
                    metadata=doc['metadata']
                ))
        
        print(f"✓ Created {len(documents)} clean documents")
        
        # Split into chunks
        print("Splitting into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✓ Created {len(chunks)} chunks")
        print(f"{'='*50}\n")
        
        return chunks
    
    def create_vector_store(self, documents: List[Document], store_name: str = "default"):
        """Create Chroma vector store from documents"""
        if not documents:
            raise ValueError("No documents to index")
        
        print(f"\n{'='*50}")
        print(f"Creating Vector Store: {store_name}")
        print(f"{'='*50}")
        
        store_path = os.path.join(self.vector_store_path, store_name)
        os.makedirs(store_path, exist_ok=True)
        
        print(f"Generating embeddings for {len(documents)} chunks...")
        print("This may take a few minutes...")
        
        # Create vector store with Ollama embeddings
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=store_path,
            collection_name=store_name
        )
        
        # Save metadata
        metadata = {
            'num_documents': len(documents),
            'sources': list(set([doc.metadata.get('source', 'unknown') for doc in documents])),
            'store_name': store_name
        }
        
        with open(os.path.join(store_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Vector store created at: {store_path}")
        print(f"✓ Indexed {len(documents)} document chunks")
        print(f"{'='*50}\n")
        
        return vector_store
    
    def load_vector_store(self, store_name: str = "default"):
        """Load existing vector store"""
        store_path = os.path.join(self.vector_store_path, store_name)
        
        if not os.path.exists(store_path):
            raise FileNotFoundError(f"Vector store not found at {store_path}")
        
        print(f"Loading vector store from {store_path}...")
        
        vector_store = Chroma(
            persist_directory=store_path,
            embedding_function=self.embeddings,
            collection_name=store_name
        )
        
        print(f"✓ Vector store loaded")
        return vector_store
    
    def add_documents_to_existing_store(self, documents: List[Document], store_name: str = "default"):
        """Add new documents to existing vector store"""
        try:
            print("Loading existing vector store...")
            vector_store = self.load_vector_store(store_name)
            
            print(f"Adding {len(documents)} new documents...")
            vector_store.add_documents(documents)
            
            print("✓ Documents added successfully")
        except FileNotFoundError:
            print("No existing store found. Creating new one...")
            vector_store = self.create_vector_store(documents, store_name)
        
        return vector_store


# Test the module
if __name__ == "__main__":
    print("Document Ingestion Module - Ready!")
    print("\nTo test:")
    print("1. Place a PDF or TXT file in data/uploads/")
    print("2. Run this test:")
    print("\n  ingestion = DocumentIngestion()")
    print("  chunks = ingestion.process_documents('data/uploads/your_file.pdf')")
    print("  vector_store = ingestion.create_vector_store(chunks, 'test_store')")