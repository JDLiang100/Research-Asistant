import os
from dotenv import load_dotenv

# Load environment variables FIRST before any other local imports
load_dotenv()

from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import search_node, research_node

def create_research_graph():
    """Creates the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("research", research_node)
    
    # Add edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "research")
    workflow.add_edge("research", END)
    
    return workflow.compile()

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is not set.")
        print("Please set your API key using: $env:GOOGLE_API_KEY = 'your_key_here'")
    else:
        # Initialize graph
        research_agent = create_research_graph()
        
        # Run research
        topic = input("Enter a research topic: ")
        
        print(f"\nSearching for information about: {topic}...\n")
        
        inputs = {"topic": topic}
        for output in research_agent.stream(inputs):
            # Output contains the results from each node
            for node_name, result in output.items():
                print(f"--- Node: {node_name} ---")
                if "current_status" in result:
                    print(f"Status: {result['current_status']}")
        
        # Print final report
        # We can get the final state from the last output or by invoking
        final_state = research_agent.invoke(inputs)
        print("\n" + "="*50)
        print("RESEARCH REPORT")
        print("="*50 + "\n")
        print(final_state["report"])
