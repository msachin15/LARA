"""
Rewrites a casual question into a tighter legal search query.

The citizen graph calls rewrite_query as its first node.
Lawyers get a prompt aimed at precedents and arguments.
Everyone else gets a prompt aimed at Indian-law search terms.
The model must answer with JSON: {"rewritten_query": "..."}.
If that JSON cannot be parsed, the question is sent to the model again
with no extra instructions, and that plain reply is used instead.
"""

import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, Annotated, List, Any
import operator
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    """Fields this node may read. Only query and role are used here."""
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str
    role: str  # "Lawyer" changes the rewrite prompt; any other value uses the citizen prompt.


def rewrite_query(state: AgentState) -> dict:
    """
    Rewrites the user's natural language query into a precise legal search query,
    tailored to the user's role.
    """
    print("---REWRITING QUERY---")
    query = state["query"]
    role = state.get("role", "Common Citizen")  # Missing role falls back to the citizen prompt.

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    if role == "Lawyer":
        # Prompt specifically for lawyers seeking precedents and arguments
        rewrite_prompt_template = """
You are a specialized legal research assistant for lawyers.
Your task is to rewrite the user's case details into a precise search query for legal precedents, relevant acts, and strong arguments.
Focus on extracting key entities like legal acts, case names, and specific arguments (e.g., "defense for X offense," "arguments against X").

Output MUST be JSON in this format:
{{ "rewritten_query": "<your query here>" }}

Original Case Details: {query}
JSON Output:
"""
    else:
        # Original prompt for common citizens
        rewrite_prompt_template = """
You are a legal research assistant specializing in Indian law.
Rewrite the user's query into a precise, legally-focused search query.

Output MUST be JSON in this format:
{{ "rewritten_query": "<your query here>" }}

Original Query: {query}
JSON Output:
"""

    rewrite_prompt = PromptTemplate(
        template=rewrite_prompt_template,
        input_variables=["query"],
    )

    try:
        # Build prompt -> model -> JSON parser, then run it on the user's question.
        rewriter_chain = rewrite_prompt | llm | JsonOutputParser()
        rewritten_query = rewriter_chain.invoke({"query": query})
    except Exception as e:
        print(f"JSON parsing failed, using raw string. Error: {e}")

        # --- FIX: Pass the query string directly to the LLM's chain ---
        raw_text_chain = (
            PromptTemplate(template="{query}", input_variables=["query"]) | llm
        )

        rewritten_text = raw_text_chain.invoke({"query": query})

        # Ensure the output is a string before putting it in the dictionary
        rewritten_query = {"rewritten_query": str(rewritten_text.content)}

    print(f"Rewritten Query: {rewritten_query}")
    return rewritten_query
