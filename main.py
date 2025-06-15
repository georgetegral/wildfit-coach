import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
from _utils.rag_service import RAGService
from contextlib import asynccontextmanager
from _utils.bot import process_telegram_update, TELEGRAM_BOT_TOKEN

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG service on startup"""
    success = rag_service.load_index()
    if not success:
        raise RuntimeError("Failed to load RAG index")
    yield
    # Aquí puedes agregar código de limpieza si lo necesitas

app = FastAPI(lifespan=lifespan)

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


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Endpoint to receive Telegram webhook updates"""
    try:
        # Get the raw JSON from the request
        update_data = await request.json()

        # Validate that we have a bot token
        if not TELEGRAM_BOT_TOKEN:
            raise HTTPException(
                status_code=500, detail="Telegram bot token not configured")

        # Process the update using the bot's process function
        await process_telegram_update(update_data, TELEGRAM_BOT_TOKEN)

        return {"status": "ok"}

    except Exception as e:
        print(f"❌ Error processing Telegram webhook: {e}")
        raise HTTPException(status_code=500, detail="Error processing webhook")


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
        "llm_model": "gpt-4o-mini",
        "telegram_bot_configured": bool(TELEGRAM_BOT_TOKEN)
    }


@app.post("/set_webhook")
async def set_telegram_webhook():
    """Endpoint to manually set the Telegram webhook (for testing/setup)"""
    try:
        from _utils.bot import main
        await main()
        return {"status": "Webhook set successfully"}
    except Exception as e:
        print(f"❌ Error setting webhook: {e}")
        raise HTTPException(status_code=500, detail="Error setting webhook")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
