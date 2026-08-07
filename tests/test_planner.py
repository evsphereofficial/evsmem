"""
Tests for the planner.py EvAgent tool interfaces.

Covers:
- memory_search tool (with and without pipeline)
- ask_memory_agent tool (with and without pipeline)
- get_tool_schemas
- execute_tool dispatch
- Tool registry structure
- Error handling when pipeline is not configured
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from evsmem.memory_pipeline import MemoryPipeline


# ============================================================
# Tests without pipeline initialized
# ============================================================

class TestPlannerNoPipeline:
    def setup_method(self):
        """Reset the global pipeline before each test."""
        from evsmem import planner
        planner._pipeline = None

    def test_memory_search_no_pipeline(self):
        """memory_search should handle gracefully with error JSON."""
        from evsmem.planner import memory_search
        result = memory_search("test query")
        parsed = json.loads(result)
        # Without pipeline, it should still return valid JSON
        assert "success" in parsed
        assert "results" in parsed

    def test_ask_memory_agent_no_pipeline(self):
        """ask_memory_agent should handle missing pipeline."""
        from evsmem.planner import ask_memory_agent
        result = ask_memory_agent("test question")
        parsed = json.loads(result)
        assert "success" in parsed
        assert "answer" in parsed or "error" in parsed

    def test_tool_schemas(self):
        """get_tool_schemas should return both tool schemas."""
        from evsmem.planner import get_tool_schemas
        schemas = get_tool_schemas()
        assert len(schemas) == 2
        schema_names = [s["name"] for s in schemas]
        assert "memory_search" in schema_names
        assert "ask_memory_agent" in schema_names
        for s in schemas:
            assert "parameters" in s
            assert "description" in s

    def test_execute_tool_memory_search(self):
        """execute_tool should dispatch to memory_search."""
        from evsmem.planner import execute_tool
        result = execute_tool("memory_search", {"query": "test"})
        parsed = json.loads(result)
        assert "success" in parsed

    def test_execute_tool_unknown(self):
        """execute_tool with unknown tool should return error."""
        from evsmem.planner import execute_tool
        result = execute_tool("nonexistent", {})
        parsed = json.loads(result)
        assert not parsed["success"]
        assert "error" in parsed

    def test_execute_tool_invalid_params(self):
        """execute_tool with missing required params should handle."""
        from evsmem.planner import execute_tool
        result = execute_tool("memory_search", {})
        parsed = json.loads(result)
        # Should still return valid JSON with error
        assert "success" in parsed


# ============================================================
# Tests with mocked pipeline
# ============================================================

class TestPlannerWithMockPipeline:
    def setup_method(self):
        from evsmem import planner
        planner._pipeline = None

    def test_memory_search_with_mock_pipeline(self):
        """Test memory_search with a mocked pipeline."""
        from evsmem.planner import memory_search, set_pipeline

        mock_pipeline = MemoryPipeline(evsmem_db_path=":memory:")
        mock_pipeline.initialize()
        set_pipeline(mock_pipeline)

        result = memory_search("search query")
        parsed = json.loads(result)
        assert "success" in parsed
        assert "results" in parsed

    def test_set_pipeline_overrides_global(self):
        """set_pipeline should replace the global pipeline instance."""
        from evsmem.planner import set_pipeline, get_pipeline
        mock_pipeline = MemoryPipeline(evsmem_db_path=":memory:")
        set_pipeline(mock_pipeline)
        assert get_pipeline() is mock_pipeline


# ============================================================
# Tool Schema Tests
# ============================================================

class TestToolSchemas:
    def test_memory_search_schema_structure(self):
        from evsmem.planner import MEMORY_SEARCH_SCHEMA
        assert MEMORY_SEARCH_SCHEMA["name"] == "memory_search"
        assert "query" in MEMORY_SEARCH_SCHEMA["parameters"]["properties"]
        assert MEMORY_SEARCH_SCHEMA["parameters"]["required"] == ["query"]

    def test_ask_memory_agent_schema_structure(self):
        from evsmem.planner import ASK_MEMORY_AGENT_SCHEMA
        assert ASK_MEMORY_AGENT_SCHEMA["name"] == "ask_memory_agent"
        assert "question" in ASK_MEMORY_AGENT_SCHEMA["parameters"]["properties"]
        assert ASK_MEMORY_AGENT_SCHEMA["parameters"]["required"] == ["question"]

    def test_tool_registry_has_both_tools(self):
        from evsmem.planner import TOOL_REGISTRY
        assert set(TOOL_REGISTRY.keys()) == {"memory_search", "ask_memory_agent"}
        for name, info in TOOL_REGISTRY.items():
            assert "handler" in info
            assert "schema" in info
            assert callable(info["handler"])


# ============================================================
# Edge Cases
# ============================================================

class TestPlannerEdgeCases:
    def setup_method(self):
        from evsmem import planner
        planner._pipeline = None

    def test_memory_search_empty_query(self):
        from evsmem.planner import memory_search
        result = memory_search("")
        parsed = json.loads(result)
        assert "success" in parsed

    def test_memory_search_with_category(self):
        from evsmem.planner import memory_search
        result = memory_search("test", category="user_preference")
        parsed = json.loads(result)
        assert "success" in parsed

    def test_memory_search_with_all_params(self):
        from evsmem.planner import memory_search
        result = memory_search("test", category="long_term_memory",
                                top_k=5, threshold=0.7)
        parsed = json.loads(result)
        assert "success" in parsed

    def test_ask_memory_agent_with_context(self):
        from evsmem.planner import ask_memory_agent
        result = ask_memory_agent("question", context="additional info")
        parsed = json.loads(result)
        assert "success" in parsed

    def test_execute_tool_ask_memory_agent(self):
        from evsmem.planner import execute_tool
        result = execute_tool("ask_memory_agent",
                               {"question": "test query"})
        parsed = json.loads(result)
        assert "success" in parsed
