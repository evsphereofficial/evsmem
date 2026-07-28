"""
Shared test fixtures for evsmem tests.

Provides temporary SQLite database fixtures and mocked embedding
functions so tests run without GPU / BGE-M3 model.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from evsmem.memory_store import MemoryStore


@pytest.fixture
def temp_db():
    """Create a temporary database file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    yield tmp.name
    try:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    except PermissionError:
        pass


@pytest.fixture
def memory_store(temp_db):
    """Create a MemoryStore with a temporary database."""
    store = MemoryStore(temp_db)
    yield store
    store.close()


@pytest.fixture(autouse=True)
def auto_mock_embedding():
    """Automatically patch generate_embedding for ALL tests in this suite.

    Patches at both the source module (evsmem.embeddings) and the importer
    modules (evsmem.retrieval) which does ``from .embeddings import generate_embedding``
    at module level, creating a local reference that would bypass the direct
    ``evsmem.embeddings`` patch.
    """
    patchers = [
        patch("evsmem.embeddings.generate_embedding", return_value=[0.0] * 1024),
        patch("evsmem.retrieval.generate_embedding", return_value=[0.0] * 1024),
    ]
    for p in patchers:
        p.start()
    yield
    for p in reversed(patchers):
        p.stop()


@pytest.fixture
def mock_embedding():
    """Patch generate_embedding to return a 1024-dim zero vector."""
    with patch("evsmem.embeddings.generate_embedding") as mock:
        mock.return_value = [0.0] * 1024
        yield mock


@pytest.fixture
def mock_embedding_failure():
    """Patch generate_embedding to raise an exception (simulating no model)."""
    with patch("evsmem.embeddings.generate_embedding") as mock:
        mock.side_effect = RuntimeError("BGE-M3 model not available")
        yield mock
