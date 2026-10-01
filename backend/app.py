"""
L.A.R.A. HTTP API.

This file is the web server the React frontend talks to.
It does not run the legal research itself. It:
  1. Accepts JSON from the browser
  2. Hands the question to the agent router
  3. Saves the chat in SQLite
  4. Sends the answer back as JSON

Run it with: python app.py
Then the API is available at http://127.0.0.1:8000
Interactive docs: http://127.0.0.1:8000/docs
"""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# route_query picks the citizen or lawyer agent and runs it.
from agent.router import route_query
# These helpers read and write chat_history.db (SQLite).
from db import (
    save_thread,
    save_message,
    get_user_threads,
    get_thread_messages,
    delete_thread,
)

# ============================
# 1. FASTAPI INITIALIZATION
# ============================

app = FastAPI(
    title="L.A.R.A. Backend API",
    description="API for the Legal Analysis & Research Assistant",
    version="1.0.0",
)

# ============================
# 2. CORS CONFIGURATION (FIXED)
# ============================

# IMPORTANT:
# - 5174 is your frontend port
# - both localhost and 127.0.0.1 MUST be allowed
# - exact match is required by browser

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 3. PYDANTIC MODELS
# ============================
# These classes describe the JSON the frontend must send.
# FastAPI rejects a request automatically if a required field is missing.

class QueryRequest(BaseModel):
    """Body for POST /process_query: the user's question plus which agent to use."""
    user_query: str  # The legal question typed in the chat box.
    role: str        # "citizen" or "lawyer". Decides which agent runs.
    thread_id: str   # UUID created by the frontend so messages stay in one chat.


class QueryResponse(BaseModel):
    """JSON returned after the agent finishes."""
    final_analysis: str  # The text shown as the bot reply.
    thread_id: str       # Same id the frontend sent, echoed back.


class ChatHistoryRequest(BaseModel):
    """Body for POST /get_chat_history. user_id comes from Clerk on the frontend."""
    user_id: str


class ThreadMessagesRequest(BaseModel):
    """Body for POST /get_thread_messages. Loads one saved conversation."""
    thread_id: str


class SaveThreadRequest(BaseModel):
    """Body for POST /save_thread. Creates the sidebar entry for a new chat."""
    user_id: str
    thread_id: str
    title: str | None = None  # Optional. If omitted, db.py builds a timestamp title.


# ============================
# 4. API ENDPOINTS
# ============================

@app.get("/")
def read_root():
    """Simple health check. Open http://127.0.0.1:8000/ to see if the server is up."""
    return {"message": "LARA backend is running."}


@app.post("/process_query", response_model=QueryResponse)
async def process_legal_query(request: QueryRequest):
    """
    Main chat endpoint. Called every time the user clicks Analyze.

    Steps:
      1. Send the question to the citizen or lawyer agent.
      2. Pull the finished text out of the agent result.
      3. Store the user message and the bot reply in SQLite.
      4. Return the reply so the chat UI can display it.
    """
    try:
        # This call can take a while: the agent searches and calls the LLM.
        result = route_query(
            role=request.role,
            user_query=request.user_query,
            thread_id=request.thread_id,
        )

        # .get avoids a crash if the agent forgot to set final_analysis.
        final_analysis = result.get(
            "final_analysis",
            "Sorry, no analysis could be generated.",
        )

        # Persist both sides of this turn so history can be reopened later.
        save_message(request.thread_id, "user", request.user_query)
        save_message(request.thread_id, "bot", final_analysis)

        return QueryResponse(
            final_analysis=final_analysis,
            thread_id=request.thread_id,
        )

    except Exception as e:
        # Log the real error on the server, then tell the browser it failed.
        print("ERROR:", e)
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/get_chat_history")
async def get_chat_history(request: ChatHistoryRequest):
    """Return every saved chat for one user, newest first. Used by the history sidebar."""
    try:
        threads = get_user_threads(request.user_id)
        return {"threads": threads}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_thread_messages")
async def get_thread_messages_endpoint(request: ThreadMessagesRequest):
    """Return every message in one chat, oldest first. Used when a sidebar item is clicked."""
    try:
        messages = get_thread_messages(request.thread_id)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_thread")
async def save_thread_endpoint(request: SaveThreadRequest):
    """Create or update the sidebar row for a chat. Called after the first real user message."""
    try:
        save_thread(request.user_id, request.thread_id, request.title)
        return {"message": "Thread saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_thread/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    """Delete one chat and all of its messages. thread_id comes from the URL, not the body."""
    try:
        delete_thread(thread_id)
        return {"message": "Thread deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# 5. LOCAL SERVER ENTRY POINT
# ============================

if __name__ == "__main__":
    # Only runs when you execute this file directly (python app.py).
    # uvicorn is the server that actually listens for HTTP requests.
    import uvicorn
    uvicorn.run(
        "app:app",       # "module:variable" — the FastAPI object named app in this file
        host="0.0.0.0",  # Listen on every network interface, not only localhost
        port=8000,
        reload=True,     # Restart automatically when Python files change
    )
