import os
from dotenv import load_dotenv

# Load environment variables FIRST before any other local imports
load_dotenv()

from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    start_node,
    query_gen_node,
    search_index_node,
    grader_node,
    writer_node,
    critique_node
)

# Load environment variables
load_dotenv()

def should_continue(state: AgentState):
    """Conditional edge logic."""
    if state.get("is_sufficient", False):
        return "end"
    return "continue"

def create_research_graph():
    """Creates a sophisticated self-corrective research graph."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("start", start_node)
    workflow.add_node("planner", query_gen_node)
    workflow.add_node("searcher", search_index_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critique_node)

    # Set Entry Point
    workflow.set_entry_point("start")

    # Add Edges
    workflow.add_edge("start", "planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "grader")
    workflow.add_edge("grader", "writer")
    workflow.add_edge("writer", "critic")

    # Add Conditional Edge (The Loop)
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "continue": "planner",
            "end": END
        }
    )

    return workflow.compile()

if __name__ == "__main__":
    print("--- Advanced AI Research Agent (Self-Corrective RAG) ---")
    topic = input("Enter a research topic: ")
    
    # Initialize graph
    research_agent = create_research_graph()
    
    inputs = {"topic": topic}
    print(f"\nStarting research on: {topic}...\n")
    
    # Run research with streaming output for feedback
    for output in research_agent.stream(inputs):
        for node_name, state in output.items():
            status = state.get("current_status", "Thinking...")
            print(f"[{node_name}] {status}")
            
            if node_name == "critic":
                if not state.get("is_sufficient"):
                    print(f"  -> FEEDBACK: {state.get('critique')}")

    # Print final report
    final_state = research_agent.invoke(inputs)
    print("\n" + "="*50)
    print("FINAL RESEARCH REPORT")
    print("="*50 + "\n")
    print(final_state["report"])
    print("\n" + "="*50)
    print("DONE: Research Complete!")
