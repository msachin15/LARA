"""
Lawyer agent: the LangGraph that answers questions for legal professionals.

Flow:
  extract issues and facts
    -> plan search sub-queries
    -> retrieve from FAISS and the web
    -> draft an answer
    -> critic scores the draft (0.0 to 1.0)
    -> if the score is under 0.7 and we have tried fewer than 3 times, plan again
    -> otherwise finalize and return the draft as final_analysis

The compiled graph is exported as `lawyer_app` and called from agent/router.py.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Any
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import LawyerAgentState
from .nodes.issue_extractor import issue_extractor
from .nodes.planner import planner
from .nodes.retriever import retriever
from .nodes.drafter import drafter
from .nodes.critic import critic
from .nodes.finalizer import finalizer

load_dotenv()

# --- Configuration & Validation ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY or not GROQ_API_KEY:
    raise ValueError(
        "API keys for Tavily and Grok are not set. Please add them to your .env file."
    )

MAX_ITERATIONS = 3


# --- Decision Nodes ---
def decide_next_step(state: LawyerAgentState):
    """
    After the critic scores a draft, either stop or revise.

    Stop when the score is at least 0.7, or when we have already revised
    MAX_ITERATIONS times. Otherwise go back to the planner for another pass.
    """
    score = state.get("answer_score", 0.0)
    iterations = state.get("iterations", 0)
    print(f"---DECISION: Score {score}, Iterations {iterations}---")

    if score >= 0.7 or iterations >= MAX_ITERATIONS:
        print("---DECISION: Ending workflow---")
        return END
    else:
        print("---DECISION: Revising draft---")
        return "planner"


# --- Build the LangGraph ---
workflow = StateGraph(LawyerAgentState)

# Add nodes
workflow.add_node("issue_extractor", issue_extractor)
workflow.add_node("planner", planner)
workflow.add_node("retriever", retriever)
workflow.add_node("drafter", drafter)
workflow.add_node("critic", critic)
workflow.add_node("finalizer", finalizer)

# Define the graph flow
workflow.set_entry_point("issue_extractor")
workflow.add_edge("issue_extractor", "planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "drafter")
workflow.add_edge("drafter", "critic")

workflow.add_conditional_edges(
    "critic",
    decide_next_step,
    {
        "planner": "planner",
        END: "finalizer",
    },
)
workflow.add_edge("finalizer", END)

# In-memory checkpoint so a thread_id can resume graph state during this server process.
checkpointer = MemorySaver()
lawyer_app = workflow.compile(checkpointer=checkpointer)
