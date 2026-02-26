from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
from tools import get_search_tool, setup_vector_store
from langchain_core.messages import HumanMessage, SystemMessage
import json
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm():
    """Returns a ChatGoogleGenerativeAI instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
    
    # Debug print (masked for safety)
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    print(f"DEBUG: Using API Key: {masked_key}")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        google_api_key=api_key # Pass explicitly
    )

# Initialize the LLM lazily or inside functions to avoid import-time errors
_llm = None

def get_node_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

def search_node(state: AgentState):
    """Generates queries and searches using DuckDuckGo."""
    topic = state["topic"]
    
    # 1. Generate queries
    system_prompt = "You are a research assistant. Generate 3 search queries to find comprehensive information about the given topic. Return them as a JSON list of strings."
    llm = get_node_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {topic}")
        ])
    except Exception as e:
        print("\n\n" + "!"*50)
        print("GOOGLE GEMINI API ERROR")
        print("!"*50)
        print(f"\nError: {e}")
        print("\nCommon fixes:")
        print("1. Key Expired/Invalid: Get a fresh key from https://aistudio.google.com/app/apikey")
        print("2. Quota: If using free tier, wait 60 seconds and try again.")
        print("!"*50 + "\n\n")
        raise e
    
    try:
        # Simple cleanup if LLM returns markdown
        content = response.content.replace("```json", "").replace("```", "").strip()
        queries = json.loads(content)
    except:
        queries = [topic] # Fallback
    
    # 2. Perform search
    search = get_search_tool()
    documents = []
    
    for query in queries:
        result = search.run(query)
        documents.append({"query": query, "content": result})
    
    return {
        "queries": queries,
        "documents": documents,
        "current_status": "Information gathered"
    }

def research_node(state: AgentState):
    """Analyzes gathered documents and synthesizes a report."""
    topic = state["topic"]
    docs = state["documents"]
    
    context = "\n\n".join([f"Source ({d['query']}): {d['content']}" for d in docs])
    
    system_prompt = "You are a research expert. Based on the provided context, write a detailed and well-structured research report about the topic. Include key facts, trends, and a summary."
    
    llm = get_node_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {topic}\n\nContext:\n{context}")
        ])
    except Exception as e:
        # Error handling is already shown above, but we can re-raise or handle here
        raise e
    
    return {
        "report": response.content,
        "current_status": "Report generated"
    }
