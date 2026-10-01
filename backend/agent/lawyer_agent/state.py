"""
Shared state passed between every node in the lawyer graph.

Each node returns a small dict. LangGraph merges that dict into this state.
Fields marked with operator.add grow (new items are appended) instead of being replaced.
"""

from typing import TypedDict, List, Any, Annotated
import operator
from langchain_core.messages import BaseMessage


class LawyerAgentState(TypedDict):
    query: str                                              # Original question from the user
    chat_history: List[BaseMessage]                         # Earlier turns loaded by the router
    issues: List[str]                                       # Legal questions pulled out of the query
    facts: List[str]                                        # Facts mentioned in the query
    sub_queries: Annotated[List[str], operator.add]         # Search strings the planner creates
    retrieved_context: Annotated[List[str], operator.add]   # Text found in FAISS and on the web
    answer_draft: str                                       # Current draft, rewritten each loop
    answer_score: float                                     # Critic score from 0.0 to 1.0
    iterations: Annotated[int, operator.add]                # How many critic passes have run
    final_analysis: str                                     # Text sent back to the API and the UI
