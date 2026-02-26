from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    # The topic of research
    topic: str
    # List of search queries generated
    queries: List[str]
    # Retrieved documents/snippets
    documents: List[dict]
    # Current stage of research
    current_status: str
    # Final synthesized report
    report: str
