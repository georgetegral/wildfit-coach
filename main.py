import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from _utils.rag_service import RAGService

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="WildFit Coach API",
    description="Ask questions about your WildFit journey",
    version="1.0.0"
)

# Initialize RAG service
rag_service = RAGService()


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG service on startup"""
    success = rag_service.load_index()
    if not success:
        raise RuntimeError("Failed to load RAG index")

# Request/Response models


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

# API endpoints


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question about WildFit and get RAG-powered answers"""
    if not rag_service.is_ready():
        raise HTTPException(status_code=500, detail="RAG service not ready")

    try:
        result = rag_service.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "WildFit Coach API is running! 🏃‍♂️", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_service_ready": rag_service.is_ready(),
        "embedding_model": "OpenAI",
        "llm_model": "gpt-4o-mini"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
