"""
SQLite storage for chats.

Two tables:
  threads  — one row per conversation (who owns it, title, timestamps)
  messages — one row per user or bot message inside a thread

The database file is backend/chat_history.db.
init_db() runs automatically when this module is imported.
"""

import sqlite3
import os
from typing import List, Dict, Any
from datetime import datetime

# Always store the db next to this file, no matter which folder you run from.
DB_PATH = os.path.join(os.path.dirname(__file__), 'chat_history.db')

def init_db():
    """Create the tables if this is the first time the app starts. Safe to call again."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # One row per chat. thread_id is the UUID the frontend generates.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # One row per message. Deleting a thread is supposed to remove its messages too,
    # but delete_thread() also deletes messages itself because SQLite foreign keys
    # are off unless PRAGMA foreign_keys = ON is set.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES threads (thread_id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()

def save_thread(user_id: str, thread_id: str, title: str = None):
    """
    Insert a new chat, or replace it if that thread_id already exists.
    If the caller does not pass a title, use the current date and time.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ? placeholders keep user text out of the SQL string (avoids SQL injection).
    cursor.execute('''
        INSERT OR REPLACE INTO threads (thread_id, user_id, title, updated_at)
        VALUES (?, ?, ?, ?)
    ''', (thread_id, user_id, title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}", datetime.now()))

    conn.commit()
    conn.close()

def save_message(thread_id: str, role: str, content: str):
    """Append one message. role is 'user' or 'bot'. timestamp is filled by SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO messages (thread_id, role, content)
        VALUES (?, ?, ?)
    ''', (thread_id, role, content))

    conn.commit()
    conn.close()

def get_user_threads(user_id: str) -> List[Dict[str, Any]]:
    """List this user's chats, most recently updated first. Used by the history sidebar."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT thread_id, title, created_at, updated_at
        FROM threads
        WHERE user_id = ?
        ORDER BY updated_at DESC
    ''', (user_id,))

    threads = []
    for row in cursor.fetchall():
        threads.append({
            'thread_id': row[0],
            'title': row[1],
            'created_at': row[2],
            'updated_at': row[3]
        })

    conn.close()
    return threads

def get_thread_messages(thread_id: str) -> List[Dict[str, Any]]:
    """Return every message in one chat, oldest first, so the UI can replay the conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT role, content, timestamp
        FROM messages
        WHERE thread_id = ?
        ORDER BY timestamp ASC
    ''', (thread_id,))

    messages = []
    for row in cursor.fetchall():
        messages.append({
            'role': row[0],
            'content': row[1],
            'timestamp': row[2]
        })

    conn.close()
    return messages

def delete_thread(thread_id: str):
    """Remove the chat row and every message that belongs to it."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('DELETE FROM threads WHERE thread_id = ?', (thread_id,))
    # Messages are deleted separately so nothing is left behind.
    cursor.execute('DELETE FROM messages WHERE thread_id = ?', (thread_id,))

    conn.commit()
    conn.close()

# Build the tables as soon as anything imports this module (app.py does).
init_db()
