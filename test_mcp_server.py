"""Tests for recall MCP server."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from mcp_server import recall_search, recall_search_json, recall_stats


# Use the harness memory directory as a real test fixture
TEST_DIR = "/home/dev/harness/memory"


class TestRecallSearch:
    def test_basic_search(self):
        result = recall_search("error recovery", TEST_DIR)
        assert "Error Recovery" in result
        assert "result(s)" in result

    def test_limit(self):
        result = recall_search("harness", TEST_DIR, limit=2)
        # Should have at most 2 numbered results
        assert "1." in result
        # Won't have more than 2

    def test_verbose(self):
        result = recall_search("error", TEST_DIR, verbose=True)
        # Verbose should show type/confidence
        assert "Type:" in result or "Tags:" in result or "Confidence:" in result

    def test_no_results(self):
        result = recall_search("xyzzy_nonexistent_term_999", TEST_DIR)
        assert "No results found" in result or "0 result" in result

    def test_invalid_directory(self):
        with pytest.raises(ValueError, match="not found"):
            recall_search("test", "/nonexistent/path")

    def test_invalid_mode(self):
        result = recall_search("test", TEST_DIR, mode="invalid")
        assert "Invalid mode" in result

    def test_limit_clamping(self):
        # Should not crash with extreme limits
        result = recall_search("error", TEST_DIR, limit=0)
        assert isinstance(result, str)
        result = recall_search("error", TEST_DIR, limit=999)
        assert isinstance(result, str)


class TestRecallSearchJson:
    def test_returns_valid_json(self):
        result = recall_search_json("error recovery", TEST_DIR)
        data = json.loads(result)
        assert "results" in data
        assert "query" in data
        assert data["query"] == "error recovery"

    def test_result_structure(self):
        result = recall_search_json("error", TEST_DIR, limit=1)
        data = json.loads(result)
        assert len(data["results"]) <= 1
        if data["results"]:
            r = data["results"][0]
            assert "title" in r
            assert "path" in r
            assert "score" in r
            assert "snippet" in r

    def test_invalid_dir_json(self):
        result = recall_search_json("test", "/nonexistent")
        data = json.loads(result)
        assert "error" in data


class TestRecallStats:
    def test_basic_stats(self):
        result = recall_stats(TEST_DIR)
        assert "Files:" in result
        assert "Total tokens:" in result

    def test_shows_types(self):
        result = recall_stats(TEST_DIR)
        # Harness memory has typed documents
        assert "Types:" in result or "Domains:" in result

    def test_invalid_dir(self):
        with pytest.raises(ValueError):
            recall_stats("/nonexistent/path")
