"""
Research step for the citizen agent.

perform_research runs two searches at the same time:
  - legal_database_search: local FAISS index of Indian law text
  - Tavily web search

Results are written back onto the graph state as plain text plus a sources list.
The embedding model and FAISS index are loaded once and reused (lazy globals).
"""

import os
from pathlib import Path
from typing import TypedDict, Annotated, List, Any
import operator

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch

# -------------------------
# Load environment variables
# -------------------------
backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(backend_dir / ".env")

# -------------------------
# GLOBAL LAZY-LOADED OBJECTS
# -------------------------
_embedding_model = None
_faiss_db = None

def get_embedding_model():
    """Load the sentence-transformer once. Later calls return the same object."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"  # small, fast model
        )
    return _embedding_model


def get_faiss_db():
    """Open the saved FAISS index once. Path is relative to the backend working directory."""
    global _faiss_db
    if _faiss_db is None:
        FAISS_INDEX_PATH = "data/faiss_index"
        _faiss_db = FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
    return _faiss_db


# -------------------------
# Agent State
# -------------------------
class AgentState(TypedDict):
    """Slice of graph state that this research step reads and writes."""
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str
    research_complete: bool
    chat_history: List[BaseMessage]
    sources: Annotated[List[dict], operator.add]


# -------------------------
# FAISS Legal DB Tool (LAZY SAFE)
# -------------------------
@tool
def legal_database_search(query: str) -> List[Document]:
    """
    Search against a pre-indexed FAISS vector store of Indian laws and cases.
    Returns a list of Document objects.
    """
    global faiss_db
    try:
        db = get_faiss_db()
        retrieved_docs = db.similarity_search_with_score(query, k=5)
        return [doc[0] for doc in retrieved_docs]

    except FileNotFoundError:
        return [Document(page_content="FAISS index not found.")]
    except Exception as e:
        return [Document(page_content=f"Error during legal database search: {e}")]


# -------------------------
# Research Function
# -------------------------
def perform_research(state: AgentState) -> dict:
    """Search the local legal index and the web, then return both as state updates."""
    print("---PERFORMING RESEARCH---")
    query = state["query"]

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")

    web_search_tool = TavilySearch(max_results=5, tavily_api_key=tavily_api_key)

    # RunnableParallel starts both searches together instead of one after the other.
    rag_chain = RunnableParallel(
        {
            "faiss_search_results": lambda x: legal_database_search.invoke(x["query"]),
            "web_search_results": lambda x: web_search_tool.invoke(x["query"]),
        }
    )

    results = rag_chain.invoke({"query": query})

    faiss_docs = results.get("faiss_search_results", [])
    web_results = results.get("web_search_results", [])

    sources = []
    faiss_content = ""

    for doc in faiss_docs:
        faiss_content += doc.page_content + "\n\n"
        if doc.metadata:
            sources.append({"type": "document", "metadata": doc.metadata})

    web_content = ""
    for result in web_results:
        web_content += result + "\n\n"
        sources.append({"type": "web", "content": result})

    print("---RESEARCH COMPLETE---")

    return {
        "faiss_search_results": faiss_content,
        "web_search_results": web_content,
        "sources": sources,
        "intermediate_steps": [
            f"FAISS Results: {faiss_content}",
            f"Web Results: {web_content}",
        ],
    }
