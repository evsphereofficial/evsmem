"""
evsmem — SQLite-based Honcho-compatible memory service.

No Docker, no Postgres, no Redis.

Usage:
  uv pip install evsmem
  evsmem          # starts on port 9876
  
Or from source:
  cd evsmem && uvicorn main:app --reload
  
API: same as Honcho (workspaces, sessions, peers, messages, context, conclusions)
Storage: SQLite at ~/.evsmem/evsmem.db
Embeddings: LM Studio at localhost:1234 (configurable)
"""

import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

__version__ = "0.1.0"

APP_DIR = Path.home() / ".evsmem"
DB_PATH = APP_DIR / "evsmem.db"
EMBEDDING_BASE_URL = "http://localhost:1234"
EMBEDDING_MODEL = "text-embedding-embeddinggemma-300m"
