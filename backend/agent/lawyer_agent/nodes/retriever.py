"""
Retriever node for the lawyer agent.

For every sub-query from the planner, this looks in two places:
  1. The local FAISS index of Indian law documents
  2. The public web, via Tavily

The combined text is stored on the state as retrieved_context.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch

from ..state import LawyerAgentState


def legal_database_search(query: str) -> str:
    """
    Search the local FAISS index and return the matching document text.

    The index is built offline by data/faiss_index/faiss_indexer.py.
    k=5 means we keep the 5 closest chunks. On any error, return the
    error text so the draft can still be written.
    """
    try:
        FAISS_INDEX_PATH = "data/faiss_index"
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = FAISS.load_local(
            FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
        )

        retrieved_docs = db.similarity_search_with_score(query, k=5)
        content = ""
        for doc, score in retrieved_docs:
            content += doc.page_content + "\n\n"
        return content
    except Exception as e:
        return f"Error during legal database search: {e}"


def web_search(query: str) -> str:
    """Search the web with Tavily and join the result snippets into one string."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")

    web_search_tool = TavilySearch(max_results=5, tavily_api_key=tavily_api_key)
    results = web_search_tool.invoke(query)
    content = ""
    for result in results:
        content += result + "\n\n"
    return content


def retriever(state: LawyerAgentState) -> dict:
    """Run both searches for each sub-query and return the combined snippets."""
    print("---RETRIEVING CONTEXT---")
    sub_queries = state["sub_queries"]

    retrieved_context = []
    for sub_query in sub_queries:
        print(f"Retrieving for: {sub_query}")
        faiss_result = legal_database_search(sub_query)
        web_result = web_search(sub_query)
        combined = f"Sub-query: {sub_query}\nFAISS: {faiss_result}\nWeb: {web_result}\n"
        retrieved_context.append(combined)

    return {"retrieved_context": retrieved_context}
