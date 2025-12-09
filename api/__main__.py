import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
import uvicorn

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI STUDY ASSISTANT API")
    print("="*60)
    print("Powered by: Ollama (100% FREE)")
    print("Starting server on: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")