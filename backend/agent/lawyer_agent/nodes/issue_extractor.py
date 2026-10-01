"""
Issue extractor node for the lawyer agent.

This is the first node in the graph. It turns a free-text question
into two lists: legal issues and facts. The planner uses both lists.
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


def issue_extractor(state: LawyerAgentState) -> dict:
    """Extract structured legal issues and facts from the user query."""
    print("---EXTRACTING ISSUES AND FACTS---")
    query = state["query"]

    llm = get_llm()
    prompt = PromptTemplate(
        template="""You are a legal assistant. Analyze the following legal query and extract:

1. **Legal Issues**: List the key legal issues or questions raised.
2. **Facts**: List the relevant facts mentioned in the query.

Provide the output in JSON format:
{{
    "issues": ["issue1", "issue2", ...],
    "facts": ["fact1", "fact2", ...]
}}

Query: {query}

Output:""",
        input_variables=["query"],
    )

    response = safe_invoke(llm, prompt, {"query": query})

    try:
        data = json.loads(response)
        issues = data.get("issues", [])
        facts = data.get("facts", [])
    except json.JSONDecodeError:
        # The model sometimes returns prose instead of JSON.
        # Fall back to the first sentence as a single issue and no facts.
        issues = [query.split(".")[0]] if "." in query else [query]
        facts = []

    return {"issues": issues, "facts": facts}
