"""
Handles all database connections using paths from .env.
"""
import os
import sqlite3
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

POSTS_DB_PATH = os.getenv("DATABASE_URL")
EVENT_DB_PATH = os.getenv("EVENT_DATABASE")

if not POSTS_DB_PATH:
    raise RuntimeError("DATABASE_URL not set in .env")
if not EVENT_DB_PATH:
    raise RuntimeError("EVENT_DATABASE not set in .env")

logger = logging.getLogger(__name__)


def get_posts_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the posts database."""
    logger.debug(f"Connecting to posts DB at {POSTS_DB_PATH}")
    conn = sqlite3.connect(POSTS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_event_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the event results database."""
    logger.debug(f"Connecting to event DB at {EVENT_DB_PATH}")
    conn = sqlite3.connect(EVENT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn