import os
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_search_tool():
    """Returns the DuckDuckGo search tool."""
    return DuckDuckGoSearchRun()

def setup_vector_store(collection_name="research_docs"):
    """Sets up an in-memory Qdrant vector store."""
    client = QdrantClient(":memory:")
    
    # Check if collection exists, if not create it
    # Note: In-memory always starts fresh, but good for local dev
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    return vector_store
