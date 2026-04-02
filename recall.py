#!/usr/bin/env python3
"""recall — hybrid search over markdown memory files.

Combines BM25 keyword search with ChromaDB semantic embeddings
using reciprocal rank fusion for best-of-both retrieval.

Usage:
    recall search <query> [--dir PATH] [--limit N] [--verbose] [--format json]
    recall search <query> [--dir PATH] [--mode bm25|semantic|hybrid]
    recall stats [--dir PATH]
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from datetime import date, datetime
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

        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            meta[key] = [item.strip().strip("\"'") for item in items if item.strip()]
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

    def content_hash(self) -> str:
        """Hash for change detection (re-embed only when content changes)."""
        return hashlib.md5((self.title + self.body).encode()).hexdigest()

    def search_text(self) -> str:
        """Combined text for semantic embedding."""
        parts = [self.title]
        if self.tags:
            parts.append(f"tags: {', '.join(self.tags)}")
        parts.append(self.body[:2000])  # Cap body to avoid huge embeddings
        return "\n".join(parts)


def load_document(filepath: Path) -> Optional[Document]:
    """Load a markdown file as a Document."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    meta, body = parse_frontmatter(content)

    title = meta.get("title", meta.get("name", ""))
    if not title:
        heading_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = heading_match.group(1) if heading_match else filepath.stem

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    return Document(
        path=str(filepath),
        title=str(title),
        body=body,
        meta=meta,
        tokens=tokenize(body),
        title_tokens=tokenize(str(title)),
        tags=[t.lower() for t in tags],
    )


# --- BM25 Index ---

@dataclass
class BM25Index:
    """BM25F index over a collection of documents.

    Uses BM25F (multi-field BM25) which properly handles multiple fields
    by normalizing TF per field, combining, then applying saturation once.
    This avoids double-counting saturation that naive per-field scoring causes.
    Reference: softwaredoug.com/blog/2025/09/18/bm25f-from-scratch
    """
    documents: list[Document] = field(default_factory=list)
    avg_body_len: float = 0.0
    avg_title_len: float = 0.0
    doc_count: int = 0
    # Per-field document frequency (max across fields for unified IDF)
    df: dict[str, int] = field(default_factory=dict)
    k1: float = 1.2
    b_body: float = 0.75     # Length normalization for body
    b_title: float = 0.3     # Less normalization for titles (they're short)
    w_title: float = 3.0     # Title field weight
    w_body: float = 1.0      # Body field weight
    tag_boost: float = 2.0

    def build(self, documents: list[Document]):
        self.documents = documents
        self.doc_count = len(documents)
        if self.doc_count == 0:
            return

        total_body = sum(d.token_count() for d in documents)
        total_title = sum(len(d.title_tokens) for d in documents)
        self.avg_body_len = total_body / self.doc_count if self.doc_count else 1
        self.avg_title_len = total_title / self.doc_count if self.doc_count else 1

        # Unified DF: max(df_body, df_title) per term
        df_body: dict[str, int] = Counter()
        df_title: dict[str, int] = Counter()
        for doc in documents:
            for term in set(doc.tokens):
                df_body[term] += 1
            for term in set(doc.title_tokens):
                df_title[term] += 1

        all_terms = set(df_body.keys()) | set(df_title.keys())
        self.df = {term: max(df_body.get(term, 0), df_title.get(term, 0))
                   for term in all_terms}

    def idf(self, term: str) -> float:
        """Unified IDF using max DF across fields."""
        n = self.df.get(term, 0)
        return math.log((self.doc_count - n + 0.5) / (n + 0.5) + 1)

    def _length_normalized_tf(self, tf: int, field_len: int,
                               avg_field_len: float, b: float) -> float:
        """Normalize term frequency by field length."""
        if field_len == 0:
            return 0.0
        return tf / (1 - b + b * field_len / avg_field_len)

    def score_document(self, doc: Document, query_tokens: list[str]) -> float:
        body_tf = Counter(doc.tokens)
        title_tf = Counter(doc.title_tokens)
        body_len = doc.token_count()
        title_len = len(doc.title_tokens)

        score = 0.0
        for term in query_tokens:
            # Step 1: Length-normalize TF per field
            norm_body = self._length_normalized_tf(
                body_tf.get(term, 0), body_len, self.avg_body_len, self.b_body
            )
            norm_title = self._length_normalized_tf(
                title_tf.get(term, 0), title_len, self.avg_title_len, self.b_title
            )

            # Step 2: Weighted combination of normalized TFs
            combined_tf = self.w_body * norm_body + self.w_title * norm_title

            if combined_tf == 0:
                continue

            # Step 3: Apply saturation ONCE to the combined TF
            saturated = combined_tf / (combined_tf + self.k1)

            # Step 4: Multiply by unified IDF
            score += self.idf(term) * saturated * (self.k1 + 1)

        # Tag exact match bonus (separate from BM25F)
        score += sum(self.tag_boost for qt in query_tokens if qt in doc.tags)

        return score

    def search(self, query: str, limit: int = 10) -> list[tuple[Document, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored = [(doc, self.score_document(doc, query_tokens))
                  for doc in self.documents]
        scored = [(doc, s) for doc, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


# --- Semantic Index (ChromaDB) ---

class SemanticIndex:
    """ChromaDB-backed semantic search with persistent storage."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self._client = None
        self._collection = None
        self._persist_dir = persist_dir

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import chromadb
            if self._persist_dir:
                self._persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self._persist_dir)
                )
            else:
                self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name="recall_memory",
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            return  # ChromaDB not available, graceful degradation

    def is_available(self) -> bool:
        self._ensure_client()
        return self._client is not None

    def index(self, documents: list[Document]):
        """Add/update documents in the semantic index."""
        if not self.is_available():
            return

        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            doc_id = hashlib.md5(doc.path.encode()).hexdigest()
            content_hash = doc.content_hash()

            # Check if document already indexed with same content
            try:
                existing = self._collection.get(ids=[doc_id], include=["metadatas"])
                if (existing["metadatas"] and
                        existing["metadatas"][0].get("content_hash") == content_hash):
                    continue  # Already up to date
            except Exception:
                pass

            ids.append(doc_id)
            texts.append(doc.search_text())
            metadatas.append({
                "path": doc.path,
                "title": doc.title,
                "content_hash": content_hash,
                "tags": ",".join(doc.tags) if doc.tags else "",
                "type": doc.meta.get("type", ""),
            })

        if ids:
            self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Search and return (path, distance) tuples."""
        if not self.is_available():
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(limit, self._collection.count() or 1),
            include=["metadatas", "distances"],
        )

        pairs = []
        if results["metadatas"] and results["distances"]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                pairs.append((meta["path"], dist))
        return pairs


# --- Fusion Methods ---

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using RRF.

    Each input is a list of (path, score) tuples, already sorted best-first.
    Returns merged list of (path, rrf_score) sorted by fused score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (path, _) in enumerate(ranked):
            scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank + 1)
    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return result


def _min_max_normalize(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Normalize scores to [0, 1] range using min-max scaling."""
    if not results:
        return results
    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [(path, 1.0) for path, _ in results]
    return [(path, (s - min_s) / (max_s - min_s)) for path, s in results]


def convex_combination(
    bm25_results: list[tuple[str, float]],
    semantic_results: list[tuple[str, float]],
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """Combine BM25 and semantic results using convex combination.

    Outperforms RRF and needs only one tunable parameter (alpha).
    alpha=0.5 weights both equally. alpha=0.7 favors BM25.

    Both result lists are min-max normalized to [0,1] before combining:
        final_score = alpha * bm25_score + (1 - alpha) * semantic_score

    Documents appearing in only one list get 0 for the missing score.
    """
    # Normalize both to [0, 1]
    bm25_norm = dict(_min_max_normalize(bm25_results))
    sem_norm = dict(_min_max_normalize(semantic_results))

    # Combine all paths
    all_paths = set(bm25_norm.keys()) | set(sem_norm.keys())
    fused = {}
    for path in all_paths:
        bm25_score = bm25_norm.get(path, 0.0)
        sem_score = sem_norm.get(path, 0.0)
        fused[path] = alpha * bm25_score + (1 - alpha) * sem_score

    result = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return result


# --- File Discovery ---

def find_markdown_files(directory: Path) -> list[Path]:
    """Recursively find all .md files, excluding hidden dirs and archive/."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "archive"]
        for fname in filenames:
            if fname.endswith(".md"):
                files.append(Path(root) / fname)
    return sorted(files)


# --- Search Orchestrator ---

def _parse_date(value: str) -> Optional[date]:
    """Try to parse a date string from frontmatter."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _recency_score(doc: Document, boost: float, half_life: float = 30.0) -> float:
    """Compute a recency bonus for a document based on its date metadata.

    Returns boost * exp(-age_days / half_life). Documents without a date
    field get zero bonus (no penalty either).
    """
    date_str = doc.meta.get("date", "")
    if not date_str:
        return 0.0
    doc_date = _parse_date(str(date_str))
    if doc_date is None:
        return 0.0
    age_days = max(0, (date.today() - doc_date).days)
    return boost * math.exp(-age_days / half_life)


def hybrid_search(
    documents: list[Document],
    query: str,
    limit: int = 10,
    mode: str = "hybrid",
    persist_dir: Optional[Path] = None,
    min_confidence: float = 0.0,
    recency_boost: float = 0.0,
    recency_half_life: float = 30.0,
) -> list[tuple[Document, float]]:
    """Run hybrid BM25 + semantic search with convex combination fusion."""
    # Filter by confidence if metadata available
    if min_confidence > 0:
        filtered = []
        for doc in documents:
            conf = doc.meta.get("confidence", "1.0")
            try:
                if float(conf) >= min_confidence:
                    filtered.append(doc)
            except (ValueError, TypeError):
                filtered.append(doc)  # No confidence = include
        documents = filtered

    doc_by_path = {d.path: d for d in documents}

    # BM25
    bm25_results = []
    if mode in ("bm25", "hybrid"):
        bm25 = BM25Index()
        bm25.build(documents)
        bm25_results = [(doc.path, score) for doc, score in bm25.search(query, limit * 2)]

    # Semantic
    semantic_results = []
    if mode in ("semantic", "hybrid"):
        sem = SemanticIndex(persist_dir=persist_dir)
        if sem.is_available():
            sem.index(documents)
            raw = sem.search(query, limit * 2)
            # Convert distances to scores (lower distance = better, invert for ranking)
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            semantic_results = [(path, 2.0 - dist) for path, dist in raw]
        elif mode == "semantic":
            print("Warning: ChromaDB not available, falling back to BM25", file=sys.stderr)
            bm25 = BM25Index()
            bm25.build(documents)
            bm25_results = [(doc.path, score) for doc, score in bm25.search(query, limit * 2)]

    # Fuse (convex combination by default, outperforms RRF)
    if mode == "hybrid" and bm25_results and semantic_results:
        fused = convex_combination(bm25_results, semantic_results, alpha=0.5)
    elif bm25_results:
        fused = bm25_results
    elif semantic_results:
        fused = semantic_results
    else:
        return []

    # Apply recency boost if enabled
    if recency_boost > 0:
        boosted = []
        for path, score in fused:
            if path in doc_by_path:
                bonus = _recency_score(doc_by_path[path], recency_boost, recency_half_life)
                boosted.append((path, score + bonus))
            else:
                boosted.append((path, score))
        boosted.sort(key=lambda x: x[1], reverse=True)
        fused = boosted

    results = []
    for path, score in fused[:limit]:
        if path in doc_by_path:
            results.append((doc_by_path[path], score))
    return results


# --- CLI ---

def cmd_search(args):
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = find_markdown_files(directory)
    if not files:
        print("No markdown files found.", file=sys.stderr)
        sys.exit(1)

    documents = [load_document(f) for f in files]
    documents = [d for d in documents if d]

    persist_dir = directory / ".recall_index"
    query = " ".join(args.query)
    results = hybrid_search(
        documents, query, args.limit, args.mode, persist_dir,
        min_confidence=args.min_confidence,
        recency_boost=args.recency_boost,
        recency_half_life=args.recency_half_life,
    )

    if not results:
        print("No results found.")
        return

    if args.format == "json":
        output = []
        for doc, score in results:
            output.append({
                "path": os.path.relpath(doc.path, directory),
                "title": doc.title,
                "score": round(score, 4),
                "tags": doc.tags,
                "type": doc.meta.get("type", ""),
                "snippet": doc.body[:300].replace("\n", " ").strip(),
            })
        print(json.dumps(output, indent=2))
        return

    for i, (doc, score) in enumerate(results, 1):
        rel_path = os.path.relpath(doc.path, directory)
        if args.verbose:
            print(f"\n{'─' * 60}")
            print(f"  [{i}] {doc.title}  (score: {score:.4f})")
            print(f"      {rel_path}")
            if doc.tags:
                print(f"      tags: {', '.join(doc.tags)}")
            if doc.meta.get("type"):
                print(f"      type: {doc.meta['type']}")
            snippet = doc.body[:200].replace("\n", " ").strip()
            if len(doc.body) > 200:
                snippet += "..."
            print(f"      {snippet}")
        else:
            print(f"  {score:.4f}  {rel_path}  —  {doc.title}")


def cmd_stats(args):
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
    if documents:
        print(f"Avg tokens/doc: {total_tokens / len(documents):.0f}")
    if types:
        print(f"Types: {dict(types)}")
    if all_tags:
        print(f"Tags: {', '.join(sorted(all_tags))}")

    # Check semantic index
    sem = SemanticIndex(persist_dir=directory / ".recall_index")
    if sem.is_available():
        sem._ensure_client()
        count = sem._collection.count() if sem._collection else 0
        print(f"Semantic index: {count} embeddings")
    else:
        print("Semantic index: unavailable (chromadb not installed)")


def cmd_add(args):
    """Add a new memory entry as a markdown file."""
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Generate title from content if not provided
    title = args.title or args.content[:60].strip().rstrip(".")
    # Generate filename from title
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50]
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = directory / f"{slug}.md"

    # Handle filename collisions
    counter = 1
    while filepath.exists():
        filepath = directory / f"{slug}-{counter}.md"
        counter += 1

    tags_str = ""
    if args.tags:
        tags_list = [t.strip() for t in args.tags.split(",")]
        tags_str = f"tags: [{', '.join(tags_list)}]\n"

    content = f"""---
type: {args.entry_type}
title: "{title}"
confidence: {args.confidence}
last_updated: {today}
source: {args.source}
{tags_str}---

{args.content}
"""

    filepath.write_text(content)
    print(f"Added: {filepath}")
    print(f"  title: {title}")
    print(f"  type: {args.entry_type}, confidence: {args.confidence}")


def main():
    parser = argparse.ArgumentParser(
        prog="recall",
        description="Hybrid search over markdown files (BM25 + semantic embeddings).",
    )
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser("search", help="Search files")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument("--dir", "-d", dest="directory", default=".", help="Directory to search")
    search_parser.add_argument("--limit", "-n", type=int, default=10, help="Max results")
    search_parser.add_argument("--verbose", "-v", action="store_true", help="Show details")
    search_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    search_parser.add_argument("--mode", "-m", choices=["bm25", "semantic", "hybrid"], default="hybrid", help="Search mode")
    search_parser.add_argument("--min-confidence", "-c", type=float, default=0.0, help="Minimum confidence threshold (0.0-1.0)")
    search_parser.add_argument("--recency-boost", "-r", type=float, default=0.0, help="Recency boost factor (0.0=off, try 0.1-0.5). Recent docs with date metadata score higher.")
    search_parser.add_argument("--recency-half-life", type=float, default=30.0, help="Half-life in days for recency decay (default: 30)")

    stats_parser = subparsers.add_parser("stats", help="Show index stats")
    stats_parser.add_argument("--dir", "-d", dest="directory", default=".", help="Directory")

    add_parser = subparsers.add_parser("add", help="Add a new memory entry")
    add_parser.add_argument("content", help="The content to remember")
    add_parser.add_argument("--dir", "-d", dest="directory", default=".", help="Directory to save to")
    add_parser.add_argument("--title", "-t", help="Entry title (auto-generated from content if omitted)")
    add_parser.add_argument("--type", dest="entry_type", default="fact", help="Entry type (fact, decision, observation, procedure)")
    add_parser.add_argument("--confidence", "-c", type=float, default=0.7, help="Confidence level 0.0-1.0")
    add_parser.add_argument("--tags", help="Comma-separated tags")
    add_parser.add_argument("--source", "-s", default="observation", help="Source (observation, operator, inference, session)")

    args = parser.parse_args()

    if args.command is None:
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            args.command = "search"
            args.query = sys.argv[1:]
            args.directory = "."
            args.limit = 10
            args.verbose = False
            args.format = "text"
            args.mode = "hybrid"
            args.min_confidence = 0.0
            args.recency_boost = 0.0
            args.recency_half_life = 30.0
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "add":
        cmd_add(args)


if __name__ == "__main__":
    main()
