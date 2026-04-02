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

# Index stats
python recall.py stats --dir ./notes
```

## How it works

1. **BM25** — standard TF-IDF keyword search with title boosting (3x) and tag matching (+2.0). Good for exact term matches.
2. **Semantic** — ChromaDB with built-in sentence-transformer embeddings. Understands that "login failures" relates to "authentication errors." Persistent index (`.recall_index/`) avoids re-embedding unchanged files.
3. **Reciprocal Rank Fusion** — combines both ranked lists into a single result set. Documents strong in both methods rank highest.

## Frontmatter support

```markdown
---
title: "My Note"
type: fact
tags: [python, search]
---

Note content here.
```

All frontmatter fields are optional. Title falls back to first `#` heading, then filename.

## License

MIT
