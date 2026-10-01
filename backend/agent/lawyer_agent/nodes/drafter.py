"""
Turns retrieved legal text into a draft answer.

The actual writing is done by generate_legacy_lawyer_answer,
which calls Groq with a lawyer-style report prompt.
"""

from ..state import LawyerAgentState
from ..legacy_answer import generate_legacy_lawyer_answer


def drafter(state: LawyerAgentState) -> dict:
    """Join every retrieved snippet into one string, then ask the LLM to draft the report."""
    print("---DRAFTING ANSWER---")
    query = state["query"]
    # retrieved_context is a list of strings, one per sub-query.
    retrieved_context = "\n".join(state["retrieved_context"])

    answer_draft = generate_legacy_lawyer_answer(query, retrieved_context)

    return {"answer_draft": answer_draft}
