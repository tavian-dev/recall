#!/usr/bin/env python3
"""recall MCP server — expose hybrid markdown search as MCP tools.

Run with:
    python mcp_server.py                    # stdio transport (default, for Claude Code)
    python mcp_server.py --transport http   # HTTP transport on port 3000
    fastmcp run mcp_server.py               # via FastMCP CLI

Add to Claude Code:
    claude mcp add recall -- python3 /home/dev/recall/mcp_server.py
"""

import json
import sys
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# Import recall's core functions
sys.path.insert(0, str(Path(__file__).parent))
from recall import (
    Document,
    find_markdown_files,
    hybrid_search,
    load_document,
)

mcp = FastMCP(
    name="recall",
    instructions=(
        "Use recall_search to find information in markdown knowledge bases. "
        "Specify the directory to search. Results are ranked by relevance using "
        "hybrid BM25 + semantic search with reciprocal rank fusion."
    ),
)


def _load_docs(directory: str) -> list[Document]:
    """Load all markdown documents from a directory."""
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    files = find_markdown_files(dir_path)
    docs = []
    for f in files:
        doc = load_document(f)
        if doc:
            docs.append(doc)
    return docs


def _format_results(results: list[tuple[Document, float]], verbose: bool = False) -> str:
    """Format search results as readable text."""
    if not results:
        return "No results found."

    lines = []
    for i, (doc, score) in enumerate(results, 1):
        # Snippet: first 200 chars of body
        snippet = doc.body[:200].replace("\n", " ").strip()
        if len(doc.body) > 200:
            snippet += "..."

        line = f"{i}. **{doc.title}** (score: {score:.4f})\n   Path: {doc.path}\n   {snippet}"

        if verbose:
            if doc.tags:
                line += f"\n   Tags: {', '.join(doc.tags)}"
            if doc.meta.get("type"):
                line += f"\n   Type: {doc.meta['type']}"
            if doc.meta.get("confidence"):
                line += f"\n   Confidence: {doc.meta['confidence']}"

        lines.append(line)

    return "\n\n".join(lines)


@mcp.tool(annotations={"readOnlyHint": True})
def recall_search(
    query: str,
    directory: str,
    limit: int = 10,
    mode: str = "bm25",
    min_confidence: float = 0.0,
    recency_boost: float = 0.0,
    verbose: bool = False,
) -> str:
    """Search markdown files by keyword and meaning. Returns ranked results with snippets.

    Uses hybrid BM25 keyword + semantic embedding search with reciprocal rank fusion.
    BM25 mode is fast and needs no embeddings. Hybrid mode requires ChromaDB.

    Args:
        query: Search keywords or natural language query.
        directory: Absolute path to the directory to search (e.g. "/home/dev/harness/memory").
        limit: Max results to return (1-50, default 10).
        mode: Search mode — "bm25" (keyword only, fast), "semantic" (embeddings only), or "hybrid" (both, best quality).
        min_confidence: Minimum confidence threshold from frontmatter (0.0-1.0). Documents without confidence metadata are always included.
        recency_boost: Boost recent documents (0.0=off, 0.1-0.5 recommended). Requires date metadata in frontmatter.
        verbose: Include tags, type, and confidence in results.
    """
    if limit < 1:
        limit = 1
    elif limit > 50:
        limit = 50

    if mode not in ("bm25", "semantic", "hybrid"):
        return f"Invalid mode '{mode}'. Use 'bm25', 'semantic', or 'hybrid'."

    docs = _load_docs(directory)
    if not docs:
        return f"No markdown files found in {directory}."

    results = hybrid_search(
        documents=docs,
        query=query,
        limit=limit,
        mode=mode,
        min_confidence=min_confidence,
        recency_boost=recency_boost,
    )

    header = f"Found {len(results)} result(s) searching {len(docs)} files in {directory}:"
    return header + "\n\n" + _format_results(results, verbose=verbose)


@mcp.tool(annotations={"readOnlyHint": True})
def recall_search_json(
    query: str,
    directory: str,
    limit: int = 10,
    mode: str = "bm25",
    min_confidence: float = 0.0,
    recency_boost: float = 0.0,
) -> str:
    """Search markdown files and return structured JSON results. Use recall_search for human-readable output.

    Args:
        query: Search keywords or natural language query.
        directory: Absolute path to the directory to search.
        limit: Max results (1-50).
        mode: "bm25", "semantic", or "hybrid".
        min_confidence: Minimum confidence threshold (0.0-1.0).
        recency_boost: Boost recent documents (0.0=off).
    """
    if limit < 1:
        limit = 1
    elif limit > 50:
        limit = 50

    if mode not in ("bm25", "semantic", "hybrid"):
        return json.dumps({"error": f"Invalid mode '{mode}'"})

    try:
        docs = _load_docs(directory)
    except ValueError as e:
        return json.dumps({"error": str(e), "results": []})
    if not docs:
        return json.dumps({"error": f"No markdown files found in {directory}", "results": []})

    results = hybrid_search(
        documents=docs,
        query=query,
        limit=limit,
        mode=mode,
        min_confidence=min_confidence,
        recency_boost=recency_boost,
    )

    output = {
        "query": query,
        "directory": directory,
        "total_files": len(docs),
        "results": [
            {
                "title": doc.title,
                "path": doc.path,
                "score": round(score, 4),
                "snippet": doc.body[:300].replace("\n", " ").strip(),
                "tags": doc.tags,
                "meta": {k: v for k, v in doc.meta.items() if k in ("type", "confidence", "domain", "last_updated")},
            }
            for doc, score in results
        ],
    }
    return json.dumps(output, indent=2)


@mcp.tool(annotations={"readOnlyHint": True})
def recall_stats(directory: str) -> str:
    """Show statistics about markdown files in a directory — file count, total tokens, tags, types.

    Args:
        directory: Absolute path to the directory to analyze.
    """
    docs = _load_docs(directory)
    if not docs:
        return f"No markdown files found in {directory}."

    total_tokens = sum(d.token_count() for d in docs)
    all_tags = set()
    types = set()
    domains = set()
    for d in docs:
        all_tags.update(d.tags)
        if d.meta.get("type"):
            types.add(d.meta["type"])
        if d.meta.get("domain"):
            domains.add(d.meta["domain"])

    lines = [
        f"Directory: {directory}",
        f"Files: {len(docs)}",
        f"Total tokens: {total_tokens:,}",
    ]
    if types:
        lines.append(f"Types: {', '.join(sorted(types))}")
    if domains:
        lines.append(f"Domains: {', '.join(sorted(domains))}")
    if all_tags:
        lines.append(f"Tags: {', '.join(sorted(all_tags))}")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="recall MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="http", host="0.0.0.0", port=args.port)
    else:
        mcp.run(transport="stdio")
