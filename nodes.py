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

def start_node(state: AgentState):
    """Initializes the research state."""
    return {
        "iterations": 0,
        "is_sufficient": False,
        "documents": [],
        "current_status": "Starting research..."
    }

def query_gen_node(state: AgentState):
    """Generates search queries, taking into account any critiques from previous hops."""
    topic = state["topic"]
    critique = state.get("critique", "")
    iterations = state.get("iterations", 0)
    
    prompt = f"You are a research planner. Topic: {topic}\n"
    if critique:
        prompt += f"Previous attempt feedback: {critique}\n"
    prompt += "\nGenerate up to 3 distinct search queries to find missing or detailed information. Return as a JSON list of strings."
    
    llm = get_node_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        queries = json.loads(content)
    except:
        queries = [topic]
        
    return {
        "queries": queries,
        "iterations": iterations + 1,
        "current_status": f"Planning hop {iterations + 1}"
    }

def search_index_node(state: AgentState):
    """Searches and adds new results to the document list."""
    queries = state["queries"]
    search = get_search_tool()
    
    new_docs = []
    for q in queries:
        try:
            result = search.run(q)
            new_docs.append({"query": q, "content": result, "id": f"doc_{len(state['documents']) + len(new_docs)}"})
        except:
            continue
            
    return {
        "documents": state["documents"] + new_docs,
        "current_status": "Gathering information"
    }

def grader_node(state: AgentState):
    """Filters out irrelevant documents using LLM grading."""
    docs = state["documents"]
    topic = state["topic"]
    llm = get_node_llm()
    
    relevant_docs = []
    for doc in docs:
        prompt = f"Is this content relevant to the topic '{topic}'? \nContent: {doc['content']}\nReply purely with 'YES' or 'NO'."
        res = llm.invoke([HumanMessage(content=prompt)])
        if "YES" in res.content.upper():
            relevant_docs.append(doc)
            
    return {
        "documents": relevant_docs,
        "current_status": "Validating sources"
    }

def writer_node(state: AgentState):
    """Synthesizes report with grounded citations."""
    topic = state["topic"]
    docs = state["documents"]
    
    context = "\n\n".join([f"[{d['id']}] {d['content']}" for d in docs])
    prompt = f"Write a detailed research report on '{topic}' based ONLY on the context below. Use citations like [doc_0]. \n\nContext: {context}"
    
    llm = get_node_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "report": response.content,
        "current_status": "Writing report"
    }

def critique_node(state: AgentState):
    """Evaluates if the report is complete or needs more hops."""
    topic = state["topic"]
    report = state["report"]
    iterations = state["iterations"]
    
    if iterations >= 3: # Safety cap
        return {"is_sufficient": True, "current_status": "Finalizing"}
        
    prompt = f"Critique this research report on '{topic}'. Is anything missing? If yes, specify what. If it's complete, say 'FINISH'.\n\nReport: {report}"
    
    llm = get_node_llm()
    res = llm.invoke([HumanMessage(content=prompt)])
    
    if "FINISH" in res.content.upper():
        return {"is_sufficient": True, "current_status": "Research verified"}
    else:
        return {
            "is_sufficient": False,
            "critique": res.content,
            "current_status": "Refining research"
        }
