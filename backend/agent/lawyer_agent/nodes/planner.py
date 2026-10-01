"""
Planner node for the lawyer agent.

Takes the extracted issues and facts and asks the LLM for 3 to 6
short search queries. Those queries are what the retriever looks up.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json

from ..state import LawyerAgentState

load_dotenv()


def get_llm():
    """Create a Groq chat model. Fails fast if GROQ_API_KEY is missing from .env."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(
        model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=groq_api_key
    )


def safe_invoke(llm, prompt, vars):
    """Fill the prompt, call the model, and return the text (not the raw message object)."""
    chain = prompt | llm
    result = chain.invoke(vars)
    return getattr(result, "content", str(result))


def planner(state: LawyerAgentState) -> dict:
    """Convert issues and facts into 3-6 retrieval sub-queries."""
    print("---PLANNING RETRIEVAL SUB-QUERIES---")
    issues = state["issues"]
    facts = state["facts"]
    query = state["query"]

    issues_str = "\n".join(issues)
    facts_str = "\n".join(facts)

    llm = get_llm()
    prompt = PromptTemplate(
        template="""You are a legal research planner. Based on the legal issues and facts, generate 3-6 specific sub-queries for retrieving relevant legal information from databases and web searches.

Issues:
{issues}

Facts:
{facts}

Original Query: {query}

Generate sub-queries that focus on:
- Relevant statutes and sections
- Case precedents
- Legal principles
- Recent judgments

Provide output as JSON list: ["sub_query1", "sub_query2", ...]

Sub-queries:""",
        input_variables=["issues", "facts", "query"],
    )

    response = safe_invoke(
        llm, prompt, {"issues": issues_str, "facts": facts_str, "query": query}
    )

    try:
        # The prompt asks for a JSON list. Parse it when the model obeys.
        sub_queries = json.loads(response)
        if not isinstance(sub_queries, list):
            sub_queries = [response]
    except json.JSONDecodeError:
        # If the model returned plain text, treat each non-empty line as a query.
        # Cap at 6 so one bad response cannot explode the search loop.
        sub_queries = [line.strip() for line in response.split("\n") if line.strip()][
            :6
        ]

    return {"sub_queries": sub_queries}
