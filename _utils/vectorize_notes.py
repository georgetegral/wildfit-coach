import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Use OpenAI embeddings (works great locally and in cloud)
Settings.embed_model = OpenAIEmbedding()

# 1. Cargar los archivos .md desde un directorio
documents = SimpleDirectoryReader("wildfit_notes").load_data()

# 2. Crear un índice vectorial
index = VectorStoreIndex.from_documents(documents)

# 3. Guardar el índice (opcional, pero recomendado para no tener que vectorizar cada vez)
index.storage_context.persist(persist_dir="storage")

print("¡Notas cargadas y vectorizadas!")