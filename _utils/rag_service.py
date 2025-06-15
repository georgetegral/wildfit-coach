import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from typing import List, Optional

class RAGService:
    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self._configure_llama_index()
    
    def _configure_llama_index(self):
        """Configure LlamaIndex with system prompt and models"""
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
    
    def load_index(self, persist_dir: str = "storage") -> bool:
        """Load the vector index from storage"""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
            print("✅ Vector index loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            print("💡 Make sure to run vectorize_notes.py first!")
            return False
    
    def query(self, question: str) -> dict:
        """Query the RAG system and return response with sources"""
        if not self.index:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Create query engine with only top results
        query_engine = self.index.as_query_engine(
            similarity_top_k=2,
            response_mode="tree_summarize"
        )
        
        # Query the index
        response = query_engine.query(question)
        
        # Extract source file names
        sources = []
        for node in response.source_nodes:
            source_name = node.metadata.get("file_name", "Unknown source")
            if source_name not in sources:
                sources.append(source_name)
        
        # Debug logging
        print(f"\n🤔 Question: {question}")
        print(f"🤖 Generated Answer: {response.response}")
        print(f"📚 Sources used: {sources}")
        
        return {
            "question": question,
            "answer": response.response,
            "sources": sources
        }
    
    def is_ready(self) -> bool:
        """Check if the service is ready to handle queries"""
        return self.index is not None
