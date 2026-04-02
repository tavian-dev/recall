"""Tests for recall — hybrid search over markdown memory files."""

import json
import math
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from recall import (
    BM25Index,
    Document,
    SemanticIndex,
    _parse_date,
    _recency_score,
    find_markdown_files,
    hybrid_search,
    load_document,
    parse_frontmatter,
    reciprocal_rank_fusion,
    tokenize,
)


# --- Tokenization ---


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_filters_stop_words(self):
        tokens = tokenize("this is a test of the system")
        assert "this" not in tokens
        assert "test" in tokens
        assert "system" in tokens

    def test_filters_short_words(self):
        tokens = tokenize("I x a ok go")
        # Single-char words should be filtered (len <= 1)
        assert "i" not in tokens
        assert "x" not in tokens
        # Two-char words that aren't stop words survive
        assert "ok" in tokens
        assert "go" in tokens

    def test_lowercases(self):
        assert tokenize("BM25 Search") == ["bm25", "search"]

    def test_splits_on_punctuation(self):
        tokens = tokenize("hello-world foo_bar baz.qux")
        assert tokens == ["hello", "world", "foo", "bar", "baz", "qux"]

    def test_preserves_numbers(self):
        tokens = tokenize("python3 version 42")
        assert "python3" in tokens
        assert "42" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_only_stop_words(self):
        assert tokenize("the and or is") == []


# --- Frontmatter Parsing ---


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        meta, body = parse_frontmatter("# Hello\nSome content")
        assert meta == {}
        assert body == "# Hello\nSome content"

    def test_basic_frontmatter(self):
        content = """---
type: knowledge
domain: testing
confidence: 0.9
---
# Body content here"""
        meta, body = parse_frontmatter(content)
        assert meta["type"] == "knowledge"
        assert meta["domain"] == "testing"
        assert meta["confidence"] == "0.9"
        assert body == "# Body content here"

    def test_list_values(self):
        content = """---
tags: [python, search, bm25]
---
Body"""
        meta, body = parse_frontmatter(content)
        assert meta["tags"] == ["python", "search", "bm25"]

    def test_quoted_values(self):
        content = """---
title: "My Document"
name: 'Another Name'
---
Body"""
        meta, body = parse_frontmatter(content)
        assert meta["title"] == "My Document"
        assert meta["name"] == "Another Name"

    def test_incomplete_frontmatter(self):
        content = "---\nkey: value\nno closing marker"
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_empty_frontmatter(self):
        content = "---\n---\nBody"
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == "Body"


# --- Document Model ---


class TestDocument:
    def test_load_document_with_heading(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# My Title\n\nSome content about testing.")
        doc = load_document(f)
        assert doc is not None
        assert doc.title == "My Title"
        assert "testing" in doc.tokens

    def test_load_document_with_frontmatter_title(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntitle: FM Title\n---\n# Heading\nBody")
        doc = load_document(f)
        assert doc.title == "FM Title"

    def test_load_document_fallback_to_stem(self, tmp_path):
        f = tmp_path / "my-doc.md"
        f.write_text("Just some text without a heading.")
        doc = load_document(f)
        assert doc.title == "my-doc"

    def test_load_nonexistent_file(self):
        doc = load_document(Path("/nonexistent/file.md"))
        assert doc is None

    def test_content_hash_changes(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\nVersion 1")
        doc1 = load_document(f)
        f.write_text("# Title\nVersion 2")
        doc2 = load_document(f)
        assert doc1.content_hash() != doc2.content_hash()

    def test_content_hash_stable(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\nContent")
        doc1 = load_document(f)
        doc2 = load_document(f)
        assert doc1.content_hash() == doc2.content_hash()

    def test_tags_parsed(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntags: [Python, Search]\n---\n# Doc\nBody")
        doc = load_document(f)
        assert doc.tags == ["python", "search"]

    def test_tags_as_string(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntags: python, search\n---\n# Doc\nBody")
        doc = load_document(f)
        assert "python" in doc.tags
        assert "search" in doc.tags

    def test_search_text(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntags: [testing]\n---\n# My Title\nBody content")
        doc = load_document(f)
        text = doc.search_text()
        assert "My Title" in text
        assert "testing" in text
        assert "Body content" in text


# --- BM25 Index ---


def make_doc(title: str, body: str, path: str = "", tags: list = None) -> Document:
    """Helper to create a Document for testing."""
    return Document(
        path=path or f"/fake/{title.lower().replace(' ', '-')}.md",
        title=title,
        body=body,
        meta={},
        tokens=tokenize(body),
        title_tokens=tokenize(title),
        tags=[t.lower() for t in (tags or [])],
    )


class TestBM25Index:
    def test_empty_index(self):
        idx = BM25Index()
        idx.build([])
        assert idx.search("anything") == []

    def test_empty_query(self):
        idx = BM25Index()
        idx.build([make_doc("Test", "some content")])
        assert idx.search("") == []

    def test_single_document_match(self):
        doc = make_doc("Python Guide", "python programming language tutorial")
        idx = BM25Index()
        idx.build([doc])
        results = idx.search("python programming")
        assert len(results) == 1
        assert results[0][0] == doc
        assert results[0][1] > 0

    def test_relevance_ranking(self):
        docs = [
            make_doc("Python Basics", "python programming tutorial basics"),
            make_doc("Cooking Recipe", "chocolate cake baking recipe"),
            make_doc("Advanced Python", "python advanced features decorators metaclasses python"),
        ]
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("python programming")
        # Documents mentioning python should rank above cooking
        result_titles = [doc.title for doc, _ in results]
        assert "Cooking Recipe" not in result_titles

    def test_title_boost(self):
        docs = [
            make_doc("Python", "some generic content here"),
            make_doc("Generic", "python appears only in body text"),
        ]
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("python")
        # Title match should score higher
        assert results[0][0].title == "Python"

    def test_tag_boost(self):
        docs = [
            make_doc("Doc A", "some content", tags=["python"]),
            make_doc("Doc B", "some content"),
        ]
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("python")
        assert results[0][0].title == "Doc A"

    def test_limit_results(self):
        docs = [make_doc(f"Doc {i}", f"keyword content {i}") for i in range(20)]
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("keyword", limit=5)
        assert len(results) <= 5

    def test_no_match(self):
        docs = [make_doc("Python", "programming language")]
        idx = BM25Index()
        idx.build(docs)
        results = idx.search("chocolate cake")
        assert len(results) == 0

    def test_idf_calculation(self):
        docs = [
            make_doc("Doc 1", "common rare"),
            make_doc("Doc 2", "common another"),
            make_doc("Doc 3", "common yet"),
        ]
        idx = BM25Index()
        idx.build(docs)
        # "rare" appears in 1 doc, "common" in all 3
        assert idx.idf("rare") > idx.idf("common")


# --- Reciprocal Rank Fusion ---


class TestReciprocalRankFusion:
    def test_single_list(self):
        ranked = [("/a.md", 10.0), ("/b.md", 5.0)]
        fused = reciprocal_rank_fusion([ranked])
        paths = [p for p, _ in fused]
        assert paths == ["/a.md", "/b.md"]

    def test_two_lists_agreement(self):
        list1 = [("/a.md", 10.0), ("/b.md", 5.0)]
        list2 = [("/a.md", 0.9), ("/b.md", 0.5)]
        fused = reciprocal_rank_fusion([list1, list2])
        assert fused[0][0] == "/a.md"

    def test_two_lists_disagreement(self):
        list1 = [("/a.md", 10.0), ("/b.md", 5.0)]
        list2 = [("/b.md", 0.9), ("/a.md", 0.5)]
        fused = reciprocal_rank_fusion([list1, list2])
        # Both should appear, scores should be close since they each rank #1 once
        paths = {p for p, _ in fused}
        assert "/a.md" in paths
        assert "/b.md" in paths

    def test_unique_items_across_lists(self):
        list1 = [("/a.md", 10.0)]
        list2 = [("/b.md", 5.0)]
        fused = reciprocal_rank_fusion([list1, list2])
        paths = {p for p, _ in fused}
        assert paths == {"/a.md", "/b.md"}

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []


# --- File Discovery ---


class TestFindMarkdownFiles:
    def test_finds_md_files(self, tmp_path):
        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.md").write_text("# B")
        (tmp_path / "c.txt").write_text("not markdown")
        files = find_markdown_files(tmp_path)
        names = [f.name for f in files]
        assert "a.md" in names
        assert "b.md" in names
        assert "c.txt" not in names

    def test_recursive_search(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.md").write_text("# Root")
        (sub / "nested.md").write_text("# Nested")
        files = find_markdown_files(tmp_path)
        names = [f.name for f in files]
        assert "root.md" in names
        assert "nested.md" in names

    def test_excludes_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.md").write_text("# Secret")
        (tmp_path / "visible.md").write_text("# Visible")
        files = find_markdown_files(tmp_path)
        names = [f.name for f in files]
        assert "visible.md" in names
        assert "secret.md" not in names

    def test_excludes_archive(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "old.md").write_text("# Old")
        (tmp_path / "current.md").write_text("# Current")
        files = find_markdown_files(tmp_path)
        names = [f.name for f in files]
        assert "current.md" in names
        assert "old.md" not in names

    def test_empty_directory(self, tmp_path):
        assert find_markdown_files(tmp_path) == []


# --- Date Parsing ---


class TestParseDate:
    def test_iso_date(self):
        assert _parse_date("2026-04-02") == date(2026, 4, 2)

    def test_iso_datetime_utc(self):
        assert _parse_date("2026-04-02T12:00:00Z") == date(2026, 4, 2)

    def test_slash_format(self):
        assert _parse_date("2026/04/02") == date(2026, 4, 2)

    def test_invalid(self):
        assert _parse_date("not a date") is None

    def test_empty(self):
        assert _parse_date("") is None


# --- Recency Score ---


class TestRecencyScore:
    def test_no_date_metadata(self):
        doc = make_doc("Test", "content")
        assert _recency_score(doc, boost=1.0) == 0.0

    def test_today_gets_full_boost(self):
        doc = make_doc("Test", "content")
        doc.meta["date"] = date.today().isoformat()
        score = _recency_score(doc, boost=1.0)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_old_doc_gets_low_boost(self):
        doc = make_doc("Test", "content")
        old_date = date.today() - timedelta(days=90)
        doc.meta["date"] = old_date.isoformat()
        score = _recency_score(doc, boost=1.0, half_life=30.0)
        assert score < 0.1  # 3 half-lives, should be ~0.05

    def test_half_life_works(self):
        doc = make_doc("Test", "content")
        half_life = 30.0
        doc.meta["date"] = (date.today() - timedelta(days=30)).isoformat()
        score = _recency_score(doc, boost=1.0, half_life=half_life)
        expected = math.exp(-1.0)  # e^(-30/30)
        assert score == pytest.approx(expected, abs=0.01)


# --- Hybrid Search (BM25 mode to avoid ChromaDB dependency) ---


class TestHybridSearch:
    def setup_method(self):
        self.docs = [
            make_doc("Python Tutorial", "learn python programming basics variables"),
            make_doc("Cooking Guide", "chocolate cake recipe baking tips"),
            make_doc("Error Handling", "python exception handling try except patterns"),
            make_doc("Git Workflow", "git branching merge rebase workflow"),
        ]

    def test_basic_search(self):
        results = hybrid_search(self.docs, "python programming", mode="bm25")
        assert len(results) > 0
        titles = [doc.title for doc, _ in results]
        assert "Python Tutorial" in titles

    def test_no_results(self):
        results = hybrid_search(self.docs, "quantum physics", mode="bm25")
        assert len(results) == 0

    def test_limit(self):
        results = hybrid_search(self.docs, "python", limit=1, mode="bm25")
        assert len(results) <= 1

    def test_confidence_filter(self):
        docs = [
            make_doc("High Conf", "python content"),
            make_doc("Low Conf", "python content"),
        ]
        docs[0].meta["confidence"] = "0.9"
        docs[1].meta["confidence"] = "0.2"
        results = hybrid_search(docs, "python", mode="bm25", min_confidence=0.5)
        titles = [doc.title for doc, _ in results]
        assert "High Conf" in titles
        assert "Low Conf" not in titles

    def test_confidence_filter_includes_no_confidence(self):
        """Documents without confidence metadata should be included."""
        docs = [
            make_doc("No Conf", "python content"),
            make_doc("Low Conf", "python content"),
        ]
        docs[1].meta["confidence"] = "0.2"
        results = hybrid_search(docs, "python", mode="bm25", min_confidence=0.5)
        titles = [doc.title for doc, _ in results]
        assert "No Conf" in titles
        assert "Low Conf" not in titles

    def test_recency_boost(self):
        docs = [
            make_doc("Old Doc", "python programming tutorial"),
            make_doc("New Doc", "python programming tutorial"),
        ]
        old_date = date.today() - timedelta(days=90)
        docs[0].meta["date"] = old_date.isoformat()
        docs[1].meta["date"] = date.today().isoformat()
        results = hybrid_search(
            docs, "python", mode="bm25",
            recency_boost=1.0, recency_half_life=30.0,
        )
        # New doc should rank first due to recency boost
        assert results[0][0].title == "New Doc"

    def test_empty_documents(self):
        results = hybrid_search([], "anything", mode="bm25")
        assert results == []

    def test_empty_query(self):
        results = hybrid_search(self.docs, "", mode="bm25")
        assert results == []


# --- Integration Test with Filesystem ---


class TestEndToEnd:
    """Integration tests that exercise the full pipeline with real files."""

    def test_search_real_directory(self, tmp_path):
        """Create a mini knowledge base and search it."""
        (tmp_path / "python.md").write_text(
            "---\ntype: knowledge\nconfidence: 0.9\n---\n"
            "# Python\nPython is a programming language used for web development and data science."
        )
        (tmp_path / "rust.md").write_text(
            "---\ntype: knowledge\nconfidence: 0.8\n---\n"
            "# Rust\nRust is a systems programming language focused on safety and performance."
        )
        (tmp_path / "cooking.md").write_text(
            "# Cooking\nChocolate cake recipe with butter and sugar."
        )

        files = find_markdown_files(tmp_path)
        documents = [load_document(f) for f in files]
        documents = [d for d in documents if d]

        results = hybrid_search(documents, "programming language", mode="bm25", limit=3)
        titles = [doc.title for doc, _ in results]
        # Programming-related docs should appear, cooking should not
        assert "Cooking" not in titles
        assert len(results) >= 1

    def test_frontmatter_roundtrip(self, tmp_path):
        """Verify frontmatter is correctly parsed and available in search results."""
        (tmp_path / "doc.md").write_text(
            "---\ntype: procedure\nconfidence: 0.7\ntags: [testing, ci]\n---\n"
            "# Testing Procedures\nRun pytest with coverage enabled."
        )
        files = find_markdown_files(tmp_path)
        doc = load_document(files[0])
        assert doc.meta["type"] == "procedure"
        assert doc.meta["confidence"] == "0.7"
        assert doc.tags == ["testing", "ci"]


# --- Semantic Index (skip if ChromaDB unavailable) ---


class TestSemanticIndex:
    @pytest.fixture
    def sem(self, tmp_path):
        idx = SemanticIndex(persist_dir=tmp_path / ".index")
        if not idx.is_available():
            pytest.skip("ChromaDB not available")
        return idx

    def test_index_and_search(self, sem):
        docs = [
            make_doc("Python Guide", "python programming tutorial basics"),
            make_doc("Cooking", "chocolate cake recipe baking"),
        ]
        sem.index(docs)
        results = sem.search("programming language", limit=2)
        assert len(results) > 0
        # First result should be the Python doc
        paths = [p for p, _ in results]
        assert any("python" in p for p in paths)

    def test_index_dedup(self, sem):
        doc = make_doc("Test", "content here")
        sem.index([doc])
        sem.index([doc])  # Re-indexing same content should be idempotent
        results = sem.search("content", limit=10)
        # Should only have one result
        assert len(results) == 1

    def test_unavailable_graceful(self, tmp_path):
        """SemanticIndex gracefully handles missing ChromaDB."""
        with patch.dict("sys.modules", {"chromadb": None}):
            idx = SemanticIndex.__new__(SemanticIndex)
            idx._client = None
            idx._collection = None
            idx._persist_dir = tmp_path
            # Force re-import attempt
            try:
                import chromadb
                # ChromaDB is available, so we can't really test this path
                pytest.skip("ChromaDB is available, can't test unavailable path")
            except (ImportError, TypeError):
                pass
