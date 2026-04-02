#!/usr/bin/env python3
"""recall — search markdown memory files using BM25 ranking.

Usage:
    recall <query> [directory] [--limit N] [--verbose]
    recall index [directory]
    recall stats [directory]

Indexes markdown files with optional YAML frontmatter and returns
ranked results. No external dependencies — pure stdlib.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --- Tokenization ---

STOP_WORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "how",
    "not", "no", "can", "do", "does", "did", "would", "could", "should",
    "i", "me", "my", "we", "our", "you", "your",
])


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, filter short words and stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOP_WORDS]


# --- Frontmatter Parsing ---

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML-ish frontmatter and body from markdown content."""
    if not content.startswith("---"):
        return {}, content

    end = content.find("---", 3)
    if end == -1:
        return {}, content

    frontmatter_str = content[3:end].strip()
    body = content[end + 3:].strip()

    # Simple YAML-ish parser (handles key: value, key: [list])
    meta = {}
    for line in frontmatter_str.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        # Handle [list] syntax
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            meta[key] = [item.strip().strip("\"'") for item in items if item.strip()]
        # Handle quoted strings
        elif (value.startswith('"') and value.endswith('"')) or \
             (value.startswith("'") and value.endswith("'")):
            meta[key] = value[1:-1]
        else:
            meta[key] = value

    return meta, body


# --- Document Model ---

@dataclass
class Document:
    path: str
    title: str
    body: str
    meta: dict
    tokens: list[str] = field(default_factory=list)
    title_tokens: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def token_count(self) -> int:
        return len(self.tokens)


def load_document(filepath: Path) -> Optional[Document]:
    """Load a markdown file as a Document."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    meta, body = parse_frontmatter(content)

    # Extract title from frontmatter or first heading
    title = meta.get("title", meta.get("name", ""))
    if not title:
        heading_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = heading_match.group(1) if heading_match else filepath.stem

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    doc = Document(
        path=str(filepath),
        title=str(title),
        body=body,
        meta=meta,
        tokens=tokenize(body),
        title_tokens=tokenize(str(title)),
        tags=[t.lower() for t in tags],
    )
    return doc


# --- BM25 Index ---

@dataclass
class BM25Index:
    """BM25 index over a collection of documents."""
    documents: list[Document] = field(default_factory=list)
    avg_doc_len: float = 0.0
    doc_count: int = 0
    df: dict[str, int] = field(default_factory=dict)  # document frequency
    k1: float = 1.2
    b: float = 0.75
    title_boost: float = 3.0
    tag_boost: float = 2.0

    def build(self, documents: list[Document]):
        """Build the index from a list of documents."""
        self.documents = documents
        self.doc_count = len(documents)
        if self.doc_count == 0:
            return

        total_tokens = sum(d.token_count() for d in documents)
        self.avg_doc_len = total_tokens / self.doc_count

        # Compute document frequencies
        self.df = Counter()
        for doc in documents:
            unique_terms = set(doc.tokens) | set(doc.title_tokens)
            for term in unique_terms:
                self.df[term] += 1

    def idf(self, term: str) -> float:
        """Inverse document frequency with smoothing."""
        n = self.df.get(term, 0)
        return math.log((self.doc_count - n + 0.5) / (n + 0.5) + 1)

    def score_document(self, doc: Document, query_tokens: list[str]) -> float:
        """Score a single document against query tokens."""
        # Body BM25
        body_tf = Counter(doc.tokens)
        doc_len = doc.token_count()
        body_score = 0.0
        for term in query_tokens:
            tf = body_tf.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            body_score += idf * numerator / denominator

        # Title BM25 (boosted)
        title_tf = Counter(doc.title_tokens)
        title_score = 0.0
        for term in query_tokens:
            tf = title_tf.get(term, 0)
            if tf == 0:
                continue
            title_score += self.idf(term) * tf  # Simplified for titles

        # Tag exact match bonus
        tag_score = sum(self.tag_boost for qt in query_tokens if qt in doc.tags)

        return body_score + self.title_boost * title_score + tag_score

    def search(self, query: str, limit: int = 10) -> list[tuple[Document, float]]:
        """Search the index and return ranked results."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for doc in self.documents:
            score = self.score_document(doc, query_tokens)
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


# --- File Discovery ---

def find_markdown_files(directory: Path) -> list[Path]:
    """Recursively find all .md files, excluding hidden dirs and archive/."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        # Skip hidden directories and archive
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "archive"]
        for fname in filenames:
            if fname.endswith(".md"):
                files.append(Path(root) / fname)
    return sorted(files)


# --- CLI ---

def cmd_search(args):
    """Search markdown files."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = find_markdown_files(directory)
    if not files:
        print("No markdown files found.", file=sys.stderr)
        sys.exit(1)

    documents = []
    for f in files:
        doc = load_document(f)
        if doc:
            documents.append(doc)

    index = BM25Index()
    index.build(documents)

    query = " ".join(args.query)
    results = index.search(query, limit=args.limit)

    if not results:
        print("No results found.")
        return

    for i, (doc, score) in enumerate(results, 1):
        rel_path = os.path.relpath(doc.path, directory)
        if args.verbose:
            print(f"\n{'─' * 60}")
            print(f"  [{i}] {doc.title}  (score: {score:.3f})")
            print(f"      {rel_path}")
            if doc.tags:
                print(f"      tags: {', '.join(doc.tags)}")
            if doc.meta.get("type"):
                print(f"      type: {doc.meta['type']}")
            # Show snippet
            snippet = doc.body[:200].replace("\n", " ").strip()
            if len(doc.body) > 200:
                snippet += "..."
            print(f"      {snippet}")
        else:
            print(f"  {score:6.3f}  {rel_path}  —  {doc.title}")


def cmd_stats(args):
    """Show index statistics."""
    directory = Path(args.directory)
    files = find_markdown_files(directory)
    documents = [load_document(f) for f in files]
    documents = [d for d in documents if d]

    total_tokens = sum(d.token_count() for d in documents)
    all_tags = set()
    types = Counter()
    for d in documents:
        all_tags.update(d.tags)
        if d.meta.get("type"):
            types[d.meta["type"]] += 1

    print(f"Directory: {directory}")
    print(f"Documents: {len(documents)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Avg tokens/doc: {total_tokens / len(documents):.0f}" if documents else "")
    if types:
        print(f"Types: {dict(types)}")
    if all_tags:
        print(f"Tags: {', '.join(sorted(all_tags))}")


def main():
    parser = argparse.ArgumentParser(
        prog="recall",
        description="Search markdown memory files using BM25 ranking.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Search (default)
    search_parser = subparsers.add_parser("search", help="Search files")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument("--dir", "-d", dest="directory", default=".", help="Directory to search")
    search_parser.add_argument("--limit", "-n", type=int, default=10, help="Max results")
    search_parser.add_argument("--verbose", "-v", action="store_true", help="Show details")

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show index stats")
    stats_parser.add_argument("--dir", "-d", dest="directory", default=".", help="Directory")

    args = parser.parse_args()

    # Default to search if no subcommand but args look like a query
    if args.command is None:
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Treat bare args as search query
            args.command = "search"
            args.query = sys.argv[1:]
            args.directory = "."
            args.limit = 10
            args.verbose = False
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
