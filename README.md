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

1. **BM25** — standard TF-IDF keyword search with title boosting (3x) and tag matching (+2.0). Good for exact term matches.
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

## License

MIT
