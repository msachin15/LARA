"""
Last step of the lawyer graph.

Copies the accepted draft into final_analysis, which is the field
the API reads and shows in the chat.
"""

from ..state import LawyerAgentState


def finalizer(state: LawyerAgentState) -> dict:
    """Promote the current draft to the answer the user will see."""
    print("---FINALIZING ANALYSIS---")
    return {"final_analysis": state["answer_draft"]}
