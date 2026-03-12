from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    # The topic of research
    topic: str
    # List of search queries generated
    queries: List[str]
    # Retrieved documents/snippets
    documents: List[dict]
    # The synthesized research report
    report: str
    # Feedback from the critique node
    critique: str
    # Current iteration count to avoid infinite loops
    iterations: int
    # Flag to exit the multi-hop loop
    is_sufficient: bool
    # Internal status tracking
    current_status: str
