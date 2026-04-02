# recall

Hybrid search over markdown files — BM25 keywords + semantic embeddings, fused with reciprocal rank fusion.

Built for searching personal knowledge bases, zettelkasten notes, and agent memory files.

## Install

Requires Python 3.10+ and ChromaDB:

```bash
pip install chromadb
```

Falls back gracefully to BM25-only if ChromaDB isn't installed.

## Usage

```bash
# Hybrid search (BM25 + semantic, default)
python recall.py search "query terms" --dir ./notes

# BM25 only (keyword matching)
python recall.py search "query" --dir ./notes --mode bm25

# Semantic only (embedding similarity)
python recall.py search "query" --dir ./notes --mode semantic

# JSON output (for programmatic use)
python recall.py search "query" --dir ./notes --format json

# Verbose (shows snippets, tags, types)
python recall.py search "query" --dir ./notes -v

# Boost recent documents (useful for journals/logs)
python recall.py search "query" --dir ./notes --recency-boost 0.3

# Custom decay rate (default half-life: 30 days)
python recall.py search "query" --dir ./notes -r 0.3 --recency-half-life 14

# Filter by confidence metadata
python recall.py search "query" --dir ./notes --min-confidence 0.7

# Index stats
python recall.py stats --dir ./notes
```

## How it works

1. **BM25F** — multi-field BM25 that properly handles title + body scoring. Normalizes term frequency per field, combines with configurable weights, then applies saturation once (avoiding the double-counting problem of naive per-field BM25). Tag exact matching adds a bonus.
2. **Semantic** — ChromaDB with built-in sentence-transformer embeddings. Understands that "login failures" relates to "authentication errors." Persistent index (`.recall_index/`) avoids re-embedding unchanged files.
3. **Reciprocal Rank Fusion** — combines both ranked lists into a single result set. Documents strong in both methods rank highest.

## Recency boost

Documents with a `date` field in frontmatter can be boosted based on how recent they are. The boost uses exponential decay:

```
bonus = boost_factor * exp(-age_days / half_life)
```

A document from today gets the full boost. After `half_life` days (~30 by default), the bonus drops to ~37% of the original. This nudges recent documents up in results without overwhelming relevance.

## Frontmatter support

```markdown
---
title: "My Note"
type: fact
tags: [python, search]
date: 2026-04-02
confidence: 0.9
---

Note content here.
```

All frontmatter fields are optional. Title falls back to first `#` heading, then filename. `date` enables recency boosting. `confidence` enables filtering with `--min-confidence`.

## Adding memories

```bash
# Add a fact
recall add "Git PAT tokens expire after 30 days" --dir ./notes --type fact --confidence 0.8

# Add with tags
recall add "BM25 scores are not normalized" --dir ./notes --tags "search,bm25" --source observation

# Add a decision
recall add "Using RRF over learned fusion for now" --dir ./notes --type decision --confidence 0.7
```

Creates a markdown file with YAML frontmatter (type, confidence, source, tags, date). Auto-generates filename from content.

## MCP Server

recall includes an MCP (Model Context Protocol) server for integration with Claude Code and other MCP-compatible tools.

```bash
# Install FastMCP
pip install fastmcp

# Register with Claude Code
claude mcp add recall -- python3 /path/to/recall/mcp_server.py

# Or run standalone
python mcp_server.py
```

### Tools exposed

- **recall_search** — search with readable text output
- **recall_search_json** — search with structured JSON output
- **recall_stats** — index statistics

Each tool accepts `query`, `directory`, `limit`, and `mode` (bm25/semantic/hybrid) parameters.

## Testing

```bash
pip install pytest
python -m pytest test_recall.py -v
```

77 tests covering tokenization, frontmatter parsing, BM25 ranking, reciprocal rank fusion, file discovery, recency scoring, hybrid search, semantic index, and MCP server tools.

## License

MIT
