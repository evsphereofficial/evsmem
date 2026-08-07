#!/usr/bin/env python3
"""
verify_untouched.py - read-only integrity checksums for the evsmem DB.

Purpose
-------
Computes a deterministic SHA-256 checksum over the FULL contents of the
`agents` and `skills` tables, plus their row counts. This is the "before"
snapshot to run BEFORE cleanup_memories.py and the "after" snapshot to run
AFTER it: if AGENTS_CHECKSUM and SKILLS_CHECKSUM are identical before and
after the cleanup, the cleanup provably did not touch those tables.

Guarantees
----------
* READ-ONLY: the database is opened via the SQLite URI "file:<path>?mode=ro",
  so the connection can never write - even on error. No DDL, no DML.
* STDLIB-ONLY: uses only `sqlite3`, `hashlib`, `os` and `sys`.
* STREAMING: rows are read with cursor.fetchmany() so memory use is bounded
  regardless of table size.
* DETERMINISTIC: rows are read in rowid order (stable insertion order) and
  serialized with fixed separators, so an unchanged table always yields the
  same checksum. The schema (column list) and row count are mixed into the
  hash so structural changes are also detected.

Run with:
    D:\\Programming\\AiProjects\\EvAgent\\evsmem\\.venv\\Scripts\\python.exe verify_untouched.py
"""

import hashlib
import os
import sqlite3
import sys

DB_PATH = r"C:\Users\Rehan\.evsmem\evsmem.db"
TABLES = ("agents", "skills")          # the two tables that must stay untouched
CHUNK_SIZE = 1000                      # rows fetched per streaming round

# Fixed separators used when serialising rows; they are bytes that cannot
# collide with normal table content, keeping the hash deterministic.
VALUE_SEP = b"\x1f"                    # between column values of one row
ROW_SEP = b"\x1e"                      # between rows


def open_readonly(db_path):
    """Open the evsmem database strictly read-only (never writable).

    Uses the SQLite file URI with mode=ro so SQLite itself enforces
    read-only access; the connection cannot create or modify the file.
    busy_timeout is set to 10000 ms as requested.
    """
    if not os.path.isfile(db_path):
        raise RuntimeError("database file not found: %s" % db_path)
    uri = "file:%s?mode=ro" % db_path.replace("\\", "/")
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def _has_rowid(conn, table):
    """Return True if `table` has an implicit integer rowid.

    Ordinary SQLite tables always have one; WITHOUT ROWID tables do not
    (and would make "ORDER BY rowid" fail), so we probe cheaply.
    """
    try:
        conn.execute("SELECT rowid FROM %s LIMIT 1" % table).fetchone()
        return True
    except sqlite3.OperationalError:
        return False


def table_checksum(conn, table):
    """Compute (sha256_hexdigest, row_count) for one table, read-only.

    The hash input is seeded with the table name, its column list (schema
    fingerprint), and its row count, then fed the full contents of every
    row in deterministic rowid order. Each row is serialised as
    str(value) joined with VALUE_SEP; rows are joined with ROW_SEP.
    """
    # 1. Schema fingerprint: column names in table order (detects renames /
    #    ADD/DROP COLUMN even when data is byte-identical).
    cols = [row[1] for row in conn.execute("PRAGMA table_info(%s)" % table)]
    if not cols:
        raise RuntimeError("table %r not found (PRAGMA table_info empty)" % table)

    # 2. Row count (printed and mixed into the hash).
    count = conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]

    # 3. Streaming hash over all rows in stable order.
    h = hashlib.sha256()
    h.update(b"table=" + table.encode("utf-8") + b"\n")
    h.update(b"columns=" + ",".join(cols).encode("utf-8") + b"\n")
    h.update(b"count=" + str(count).encode("utf-8") + b"\n")

    order_by = "rowid" if _has_rowid(conn, table) else cols[0]
    cur = conn.execute("SELECT * FROM %s ORDER BY %s" % (table, order_by))
    while True:
        rows = cur.fetchmany(CHUNK_SIZE)
        if not rows:
            break
        for row in rows:
            serialised = VALUE_SEP.join(
                str(value).encode("utf-8", "backslashreplace") for value in row
            )
            h.update(serialised)
            h.update(ROW_SEP)

    return h.hexdigest(), count


def main():
    print("=" * 78)
    print("evsmem INTEGRITY CHECK (read-only) - untouched-table checksums")
    print("DB: %s" % DB_PATH)
    print("=" * 78)

    try:
        conn = open_readonly(DB_PATH)
    except RuntimeError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    try:
        for table in TABLES:
            try:
                digest, count = table_checksum(conn, table)
            except sqlite3.OperationalError as exc:
                print("ERROR: cannot checksum table %r: %s" % (table, exc),
                      file=sys.stderr)
                return 1
            label = table.upper() + "_CHECKSUM"
            print("%s = sha256:%s" % (label, digest))
            print("  %s row count: %d" % (table, count))
    finally:
        conn.close()

    print("-" * 78)
    print("Connection opened mode=ro; no data was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
