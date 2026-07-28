"""
Tests for the MemoryStore SQLite persistence layer.

Covers:
- Schema initialisation (all 6 category tables + raw queue + cursor state)
- CRUD for each memory category
- Raw message queue operations
- Cursor state
- Embedding storage and retrieval
- Text search
- Similarity search (with mocked embeddings)
- Edge cases: unknown category, empty content, missing IDs
"""

import os
import pytest
import sqlite3
import tempfile

from evsmem.memory_store import MemoryStore
from evsmem.schemas import CATEGORY_TABLE_MAP


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def store():
    """Create a temporary MemoryStore for testing."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    s = MemoryStore(tmp.name)
    yield s
    # Close connection before unlinking (Windows requires this)
    s.close()
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass  # file may still be locked, that's OK for temp cleanup


@pytest.fixture
def populated_store(store):
    """Store preloaded with one memory of each category."""
    categories = {
        "hot_memory": dict(content="current task", context_id="ctx1",
                           expires_at="2026-12-31T23:59:59"),
        "user_preference": dict(content="likes Python",
                                preference_key="language",
                                preference_value="Python",
                                category="code"),
        "behavior_pattern": dict(content="frequent testing",
                                 pattern_type="testing",
                                 frequency=0.8),
        "conclusion": dict(content="user is a developer",
                           conclusion_type="occupation",
                           supporting_evidence=["mentions coding often"]),
        "long_term_memory": dict(content="user has RTX 4070",
                                 memory_type="device",
                                 tags=["gpu", "nvidia"]),
        "relationship": dict(content="EvAgent uses SQLite",
                             source_entity="EvAgent",
                             target_entity="SQLite",
                             relationship_type="uses"),
    }
    ids = {}
    for cat, fields in categories.items():
        ids[cat] = store.create_memory(cat, **fields)
    return store, ids


# ============================================================
# Schema & Initialisation
# ============================================================

class TestMemoryStoreInit:
    def test_init_creates_tables(self, store):
        """Verify all 6 category tables plus raw queue exist."""
        from evsmem.config import get_all_new_table_names

        conn = sqlite3.connect(store._db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()

        # The new evsmem-prefixed tables should all be present.
        # evsmem_schema_version is created by migration, not by MemoryStore._init_schema().
        expected_tables = [
            t for t in get_all_new_table_names()
            if t != "evsmem_schema_version"
        ]
        for tbl in expected_tables:
            assert tbl in tables, f"Missing table: {tbl}"

        # Old CATEGORY_TABLE_MAP short names should NOT be present
        for short_name in CATEGORY_TABLE_MAP.values():
            assert short_name not in tables, f"Short table name should not exist: {short_name}"

    def test_init_is_idempotent(self, store):
        """Calling _init_schema twice should not raise."""
        store._init_schema()  # second call
        assert True  # no exception

    def test_default_db_path(self):
        """Default path should be ~/.evsmem/evsmem.db."""
        from evsmem.config import EVSMEM_DB_PATH
        assert ".evsmem" in str(EVSMEM_DB_PATH)
        assert "evsmem.db" in str(EVSMEM_DB_PATH)

    def test_custom_db_path(self):
        """MemoryStore can be created with any path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        s = MemoryStore(tmp.name)
        assert s._db_path.name == os.path.basename(tmp.name)
        s.close()
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass


# ============================================================
# CRUD: Create
# ============================================================

class TestMemoryStoreCreate:
    def test_create_hot_memory(self, store):
        mid = store.create_memory("hot_memory", content="test",
                                  context_id="ctx1")
        assert mid is not None
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_create_user_preference(self, store):
        mid = store.create_memory("user_preference", content="likes Python",
                                  preference_key="language",
                                  preference_value="Python",
                                  category="code")
        assert mid is not None
        mem = store.get_memory(mid, "user_preference")
        assert mem is not None
        assert mem["preference_key"] == "language"
        assert mem["preference_value"] == "Python"
        assert mem["category"] == "code"

    def test_create_relationship(self, store):
        mid = store.create_memory("relationship", content="A uses B",
                                  source_entity="A", target_entity="B",
                                  relationship_type="uses")
        assert mid is not None

    def test_create_long_term_memory(self, store):
        mid = store.create_memory("long_term_memory", content="test fact",
                                  memory_type="general", tags=["test"])
        assert mid is not None

    def test_create_behavior_pattern(self, store):
        mid = store.create_memory("behavior_pattern", content="frequent commits",
                                  pattern_type="coding", frequency=0.9)
        assert mid is not None
        mem = store.get_memory(mid, "behavior_pattern")
        assert mem is not None
        assert abs(mem["frequency"] - 0.9) < 0.01

    def test_create_conclusion(self, store):
        mid = store.create_memory("conclusion", content="user prefers Python",
                                  conclusion_type="preference",
                                  supporting_evidence=["said so"],
                                  confidence=0.8)
        assert mid is not None
        mem = store.get_memory(mid, "conclusion")
        assert mem is not None
        assert mem["confidence"] == 0.8

    def test_create_unknown_category_raises(self, store):
        with pytest.raises(ValueError, match="Unknown memory category"):
            store.create_memory("nonexistent", content="test")

    def test_create_with_minimal_args(self, store):
        """Content is the only truly required field."""
        mid = store.create_memory("long_term_memory", content="minimal",
                                  memory_type="test")
        assert mid is not None
        mem = store.get_memory(mid, "long_term_memory")
        assert mem["content"] == "minimal"
        assert mem["confidence"] == 0.5  # default
        assert mem["importance"] == 5    # default

    def test_create_with_metadata(self, store):
        mid = store.create_memory(
            "long_term_memory", content="with meta",
            memory_type="test",
            metadata={"source": "test", "version": 1},
        )
        assert mid is not None
        mem = store.get_memory(mid, "long_term_memory")
        assert mem["metadata"]["source"] == "test"

    def test_create_with_importance_and_confidence(self, store):
        mid = store.create_memory("long_term_memory", content="important",
                                  memory_type="test",
                                  importance=9, confidence=0.95)
        mem = store.get_memory(mid, "long_term_memory")
        assert mem["importance"] == 9
        assert mem["confidence"] == 0.95


# ============================================================
# CRUD: Read
# ============================================================

class TestMemoryStoreRead:
    def test_get_memory_exists(self, populated_store):
        store, ids = populated_store
        mem = store.get_memory(ids["long_term_memory"], "long_term_memory")
        assert mem is not None
        assert mem["content"] == "user has RTX 4070"
        assert mem["memory_type"] == "device"

    def test_get_memory_not_found(self, store):
        mem = store.get_memory("nonexistent-id", "long_term_memory")
        assert mem is None

    def test_get_memory_unknown_category_raises(self, store):
        with pytest.raises(ValueError, match="Unknown memory category"):
            store.get_memory("some-id", "nonexistent")

    def test_get_memory_returns_dict_with_all_fields(self, populated_store):
        store, ids = populated_store
        mem = store.get_memory(ids["user_preference"], "user_preference")
        assert isinstance(mem, dict)
        assert "id" in mem
        assert "content" in mem
        assert "created_at" in mem
        assert "updated_at" in mem
        assert "preference_key" in mem
        assert "preference_value" in mem

    def test_get_all_memories_by_category(self, populated_store):
        store, ids = populated_store
        all_hot = store.get_all_memories_by_category("hot_memory")
        assert len(all_hot) >= 1

    def test_get_all_memories_empty_category(self, store):
        all_rel = store.get_all_memories_by_category("relationship")
        assert all_rel == []

    def test_get_all_memories_unknown_category_raises(self, store):
        with pytest.raises(ValueError, match="Unknown memory category"):
            store.get_all_memories_by_category("nonexistent")


# ============================================================
# CRUD: Update
# ============================================================

class TestMemoryStoreUpdate:
    def test_update_content(self, store):
        mid = store.create_memory("long_term_memory", content="original",
                                  memory_type="test")
        assert store.update_memory(mid, "long_term_memory",
                                    content="updated", importance=8)
        mem = store.get_memory(mid, "long_term_memory")
        assert mem["content"] == "updated"
        assert mem["importance"] == 8

    def test_update_partial(self, store):
        mid = store.create_memory("long_term_memory", content="original",
                                  memory_type="test", confidence=0.5)
        store.update_memory(mid, "long_term_memory", confidence=0.9)
        mem = store.get_memory(mid, "long_term_memory")
        assert mem["confidence"] == 0.9
        assert mem["content"] == "original"  # unchanged

    def test_update_no_changes(self, store):
        mid = store.create_memory("long_term_memory", content="test",
                                  memory_type="t")
        assert not store.update_memory(mid, "long_term_memory")  # no updates

    def test_update_nonexistent(self, store):
        assert not store.update_memory("no-such-id", "long_term_memory",
                                        content="anything")

    def test_update_unknown_category_raises(self, store):
        with pytest.raises(ValueError, match="Unknown memory category"):
            store.update_memory("some-id", "nonexistent", content="x")


# ============================================================
# CRUD: Delete
# ============================================================

class TestMemoryStoreDelete:
    def test_delete_memory(self, store):
        mid = store.create_memory("hot_memory", content="delete me",
                                  context_id="ctx1")
        assert store.delete_memory(mid, "hot_memory")
        assert store.get_memory(mid, "hot_memory") is None

    def test_delete_nonexistent(self, store):
        assert not store.delete_memory("no-such-id", "long_term_memory")

    def test_delete_unknown_category_raises(self, store):
        with pytest.raises(ValueError, match="Unknown memory category"):
            store.delete_memory("some-id", "nonexistent")

    def test_delete_then_recreate_same_id(self, store):
        """Re-creating with the same UUID should work after deletion."""
        import uuid
        mid = str(uuid.uuid4())
        store.create_memory("long_term_memory", content="first",
                            memory_type="t", id=mid)
        store.delete_memory(mid, "long_term_memory")
        store.create_memory("long_term_memory", content="second",
                            memory_type="t", id=mid)
        mem = store.get_memory(mid, "long_term_memory")
        assert mem["content"] == "second"


# ============================================================
# Raw Queue
# ============================================================

class TestRawQueue:
    def test_append_raw_message(self, store):
        mid = store.append_raw_message("conv1", "user", "Hello")
        assert isinstance(mid, int)
        assert mid > 0

    def test_create_batch(self, store):
        mid1 = store.append_raw_message("conv1", "user", "Hello")
        mid2 = store.append_raw_message("conv1", "assistant", "Hi")
        batch_id = store.create_batch("conv1", [mid1, mid2])
        assert batch_id is not None
        assert isinstance(batch_id, int)

    def test_get_unprocessed_batches(self, store):
        mid1 = store.append_raw_message("conv1", "user", "A")
        mid2 = store.append_raw_message("conv1", "assistant", "B")
        store.create_batch("conv1", [mid1, mid2])
        batches = store.get_unprocessed_batches(min_size=1)
        assert len(batches) >= 1

    def test_mark_batch_processed(self, store):
        mid1 = store.append_raw_message("conv1", "user", "X")
        mid2 = store.append_raw_message("conv1", "assistant", "Y")
        batch_id = store.create_batch("conv1", [mid1, mid2])
        store.mark_batch_processed(batch_id)
        batches = store.get_unprocessed_batches(min_size=1)
        # Our batch should no longer be in unprocessed
        assert not any(b.id == batch_id for b in batches)

    def test_get_unprocessed_batches_min_size(self, store):
        mid = store.append_raw_message("conv1", "user", "Only one")
        store.create_batch("conv1", [mid])
        # With min_size=10, batches with 1 message should not appear
        batches = store.get_unprocessed_batches(min_size=10)
        assert len(batches) == 0


# ============================================================
# Cursor
# ============================================================

class TestCursor:
    def test_cursor_initial_zero(self, store):
        assert store.get_cursor() == 0

    def test_cursor_update(self, store):
        store.update_cursor(42)
        assert store.get_cursor() == 42

    def test_cursor_multiple_updates(self, store):
        store.update_cursor(10)
        store.update_cursor(20)
        store.update_cursor(30)
        assert store.get_cursor() == 30

    def test_cursor_persists(self, store):
        """Cursor survives re-opening the DB."""
        store.update_cursor(99)
        # Create a new store instance pointing to same file
        store2 = MemoryStore(store._db_path)
        assert store2.get_cursor() == 99


# ============================================================
# Embeddings
# ============================================================

class TestEmbeddings:
    def test_store_and_get_embedding(self, store):
        mid = store.create_memory("long_term_memory", content="embed me",
                                  memory_type="test")
        emb = [0.1, 0.2, 0.3, 0.4]
        store.store_embedding(mid, "long_term_memory", emb)
        retrieved = store.get_embedding(mid, "long_term_memory")
        assert retrieved is not None
        assert len(retrieved) == 4
        assert abs(retrieved[0] - 0.1) < 0.001
        assert abs(retrieved[2] - 0.3) < 0.001

    def test_get_embedding_no_embedding(self, store):
        mid = store.create_memory("long_term_memory", content="no emb",
                                  memory_type="test")
        retrieved = store.get_embedding(mid, "long_term_memory")
        assert retrieved is None

    def test_get_embedding_nonexistent(self, store):
        retrieved = store.get_embedding("no-such-id", "long_term_memory")
        assert retrieved is None

    def test_overwrite_embedding(self, store):
        mid = store.create_memory("long_term_memory", content="overwrite",
                                  memory_type="test")
        store.store_embedding(mid, "long_term_memory", [0.1, 0.2])
        store.store_embedding(mid, "long_term_memory", [0.9, 0.8])
        retrieved = store.get_embedding(mid, "long_term_memory")
        assert abs(retrieved[0] - 0.9) < 0.001


# ============================================================
# Search
# ============================================================

class TestSearch:
    def test_search_by_text(self, store):
        store.create_memory("long_term_memory", content="python programming",
                            memory_type="skill", tags=["python"])
        store.create_memory("long_term_memory", content="javascript",
                            memory_type="skill", tags=["js"])
        results = store.search_by_text(None, "python", 10)
        assert len(results) >= 1
        assert "python" in results[0]["content"].lower()

    def test_search_by_text_category_filter(self, store):
        store.create_memory("long_term_memory", content="loves hiking",
                            memory_type="hobby")
        store.create_memory("user_preference", content="loves hiking",
                            preference_key="outdoor",
                            preference_value="hiking",
                            category="lifestyle")
        results = store.search_by_text("long_term_memory", "hiking", 10)
        assert len(results) >= 1
        for r in results:
            assert r["_category"] == "long_term_memory"

    def test_search_by_text_no_match(self, store):
        results = store.search_by_text(None, "zzzzzznonexistent", 10)
        assert results == []

    def test_search_by_text_unknown_category_raises(self, store):
        with pytest.raises(ValueError, match="Unknown memory category"):
            store.search_by_text("nonexistent", "query", 10)


# ============================================================
# Similarity Search (with mocked embedding)
# ============================================================

class TestSimilaritySearch:
    def test_similarity_search_works_with_embeddings(self, store):
        """Exercise cosine-similarity code path with stored embeddings."""
        mid1 = store.create_memory("long_term_memory", content="python coding",
                                   memory_type="skill")
        mid2 = store.create_memory("long_term_memory", content="java coding",
                                   memory_type="skill")
        store.store_embedding(mid1, "long_term_memory", [1.0, 0.0, 0.0])
        store.store_embedding(mid2, "long_term_memory", [0.9, 0.1, 0.0])
        results = store.similarity_search([1.0, 0.0, 0.0], top_k=10)
        assert len(results) >= 1
        for r in results:
            assert "_similarity" in r
            assert isinstance(r["_similarity"], float)

    def test_similarity_search_category_filter(self, store):
        mid = store.create_memory("long_term_memory", content="test",
                                  memory_type="t")
        store.store_embedding(mid, "long_term_memory", [0.5, 0.5])
        results = store.similarity_search([0.5, 0.5],
                                           memory_category="long_term_memory",
                                           top_k=10)
        assert len(results) >= 1
        for r in results:
            assert r["_category"] == "long_term_memory"

    def test_similarity_search_no_embeddings(self, store):
        """No embeddings stored → empty result list."""
        store.create_memory("long_term_memory", content="no embedding",
                            memory_type="t")
        results = store.similarity_search([1.0, 0.0], top_k=10)
        assert results == []

    def test_similarity_search_zero_vector(self, store):
        """Zero query vector should return empty."""
        results = store.similarity_search([0.0, 0.0, 0.0], top_k=10)
        assert results == []

    def test_similarity_search_unknown_category_raises(self, store):
        with pytest.raises(ValueError, match="Unknown memory category"):
            store.similarity_search([0.1, 0.2],
                                     memory_category="nonexistent")


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_content(self, store):
        mid = store.create_memory("long_term_memory", content="",
                                  memory_type="t")
        mem = store.get_memory(mid, "long_term_memory")
        assert mem is not None
        assert mem["content"] == ""

    def test_very_long_content(self, store):
        long_content = "A" * 10000
        mid = store.create_memory("long_term_memory", content=long_content,
                                  memory_type="t")
        mem = store.get_memory(mid, "long_term_memory")
        assert len(mem["content"]) == 10000

    def test_unicode_content(self, store):
        content = "Hello 世界 🌍 emoji test ✓"
        mid = store.create_memory("long_term_memory", content=content,
                                  memory_type="t")
        mem = store.get_memory(mid, "long_term_memory")
        assert mem["content"] == content

    def test_multiple_memories_same_category(self, store):
        ids = []
        for i in range(10):
            mid = store.create_memory("long_term_memory",
                                       content=f"memory {i}",
                                       memory_type="test")
            ids.append(mid)
        all_mem = store.get_all_memories_by_category("long_term_memory")
        assert len(all_mem) == 10

    def test_get_all_memories_returns_all_categories(self, populated_store):
        store, ids = populated_store
        for cat in ("hot_memory", "user_preference", "behavior_pattern",
                     "conclusion", "long_term_memory", "relationship"):
            mems = store.get_all_memories_by_category(cat)
            assert len(mems) >= 1, f"No memories in {cat}"

    def test_concurrent_store_access(self):
        """Two MemoryStore instances pointing to same file should work."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        s1 = MemoryStore(tmp.name)
        s2 = MemoryStore(tmp.name)
        mid = s1.create_memory("long_term_memory", content="shared",
                                memory_type="t")
        mem = s2.get_memory(mid, "long_term_memory")
        s1.close()
        s2.close()
        assert mem is not None
        assert mem["content"] == "shared"
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass
