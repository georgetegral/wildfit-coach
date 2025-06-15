import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from typing import List

# Load environment variables
load_dotenv()

# Configure LlamaIndex with system prompt
Settings.embed_model = OpenAIEmbedding()
Settings.llm = OpenAI(
    model="gpt-4.1-nano", 
    temperature=0.1,
    system_prompt="""
Tu nombre es Eric Edmeades, eres el fundador del programa WildFit y un coach nutricional con más de 20 años de experiencia. Eres conocido por tu enfoque directo, gracioso e ingenioso, pero siempre motivador y compasivo, te gusta hacer bromas.

Tu función es ayudar a las personas con preguntas sobre nutrición, estaciones metabólicas y el programa WildFit en general. También das apoyo emocional y motivas al usuario a seguir el programa.

Instrucciones:
- Responde de manera clara y concisa, usando un lenguaje sencillo y fácil de entender. Evita la jerga técnica a menos que sea absolutamente necesario, y explica los términos técnicos si los usas.
- Cuando respondas preguntas, prioriza la información que se encuentra en mis notas del programa WildFit. Si no encuentras la respuesta en mis notas, usa tu propio conocimiento, pero siempre aclara que estás usando información externa.
- Ofrece apoyo emocional y motivación, pero evita dar consejos médicos o psicológicos. Si la persona parece estar pasando por un momento difícil, anímala a buscar ayuda profesional.
- Responde en el idioma que se te pregunte.
"""
)

app = FastAPI(title="WildFit Coach API", description="Ask questions about your WildFit journey")

# Global variable to hold the index
index = None

@app.on_event("startup")
async def startup_event():
    """Load the vector index on startup"""
    global index
    try:
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        index = load_index_from_storage(storage_context)
        print("✅ Vector index loaded successfully")
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        print("💡 Make sure to run vectorize_notes.py first!")
        raise

class QueryRequest(BaseModel):
    question: str

class Answer(BaseModel):
    content: str
    source: str
    confidence: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question about WildFit and get RAG-powered answers"""
    try:
        if not index:
            raise HTTPException(status_code=500, detail="Vector index not loaded")
        
        # Create query engine with only 2 top results
        query_engine = index.as_query_engine(
            similarity_top_k=3,  # Solo las 3 respuestas más relevantes
            response_mode="tree_summarize"
        )
        
        # Query the index
        response = query_engine.query(request.question)
        
        # Extract source file names
        sources = []
        for node in response.source_nodes:
            source_name = node.metadata.get("file_name", "Unknown source")
            if source_name not in sources:
                sources.append(source_name)
        
        # Print to console for debugging
        print(f"\n🤔 Question: {request.question}")
        print(f"🤖 Generated Answer: {response.response}")
        print(f"📚 Sources used: {sources}")
        
        return QueryResponse(
            question=request.question,
            answer=response.response,
            sources=sources
        )
        
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "WildFit Coach API is running! 🏃‍♂️", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "embedding_model": "OpenAI",
        "llm_model": "gpt-4.1-nano",
        "version": "1.0.0",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
