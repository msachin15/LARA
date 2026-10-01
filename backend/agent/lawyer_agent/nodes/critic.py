"""
Critic node for the lawyer agent.

Scores the current draft from 0.0 to 1.0 using two signals:
  1. The LLM's own judgment of relevance, accuracy, and completeness
  2. How similar the draft is to the original question (embedding cosine similarity)

The graph uses this score to decide whether to revise or finish.
Returning iterations: 1 adds one to the running count (operator.add).
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util

from ..state import LawyerAgentState

load_dotenv()

# Loaded once when this module is imported, then reused for every score.
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


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


def critic(state: LawyerAgentState) -> dict:
    """Evaluate the draft answer and assign a score 0.0-1.0."""
    print("---CRITIC EVALUATION---")
    query = state["query"]
    answer_draft = state["answer_draft"]
    retrieved_context = "\n".join(state["retrieved_context"])

    llm = get_llm()
    eval_prompt = PromptTemplate(
        template="""Evaluate the legal answer draft on a scale of 0.0 to 1.0.

Query: {query}
Retrieved Context: {context}
Answer Draft: {answer}

Score based on:
- Relevance to query
- Use of retrieved context
- Legal accuracy
- Completeness

Output only a number between 0.0 and 1.0:""",
        input_variables=["query", "context", "answer"],
    )

    llm_score_str = safe_invoke(
        llm,
        eval_prompt,
        {"query": query, "context": retrieved_context[:2000], "answer": answer_draft},
    )
    try:
        llm_score = float(llm_score_str.strip())
        # Clamp so a model that prints 1.4 or -0.2 cannot break the 0–1 scale.
        llm_score = max(0.0, min(1.0, llm_score))
    except ValueError:
        # The model sometimes adds words around the number. Use a neutral score.
        llm_score = 0.5

    # Second signal: cosine similarity between the question and the draft.
    # 1.0 means the texts point the same direction in embedding space.
    if EMBEDDING_MODEL:
        emb_query = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
        emb_answer = EMBEDDING_MODEL.encode(answer_draft, convert_to_tensor=True)
        sim_score = util.cos_sim(emb_answer, emb_query).item()
        sim_score = max(0.0, min(1.0, sim_score))
    else:
        sim_score = 0.5

    # Equal weight on the LLM score and the similarity score.
    final_score = (llm_score + sim_score) / 2

    return {"answer_score": final_score, "iterations": 1}
