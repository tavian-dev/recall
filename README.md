# recall

Search markdown files using BM25 ranking. Zero dependencies, pure Python.

Built for searching personal knowledge bases, zettelkasten notes, and agent memory files — anywhere you have a directory of markdown files and want to find things fast.

## Usage

```bash
# Search
python recall.py search "query terms" --dir ./notes

# Verbose output (shows snippets, tags, types)
python recall.py search "query terms" --dir ./notes -v

# Limit results
python recall.py search "query" --dir ./notes -n 5

# Stats
python recall.py stats --dir ./notes
```

## Features

- **BM25 ranking** — standard information retrieval algorithm, no ML needed
- **YAML frontmatter support** — indexes `title`, `tags`, `type`, and other metadata
- **Title boosting** — matches in titles rank higher than body matches
- **Tag matching** — exact tag matches get a bonus
- **Zero dependencies** — pure Python stdlib, works everywhere
- **Fast** — indexes on every run, no persistent state needed for small collections

## Frontmatter format

```markdown
---
title: "My Note"
type: fact
tags: [python, search]
confidence: 0.9
---

Note content here.
```

All frontmatter fields are optional. Title falls back to first `#` heading, then filename.

## How scoring works

Each document is scored against the query using:
1. **BM25 on body tokens** (k1=1.2, b=0.75) — standard TF-IDF with length normalization
2. **BM25 on title tokens** (3x boost) — title matches are weighted heavily
3. **Tag exact match** (+2.0 per matching tag) — direct tag hits

Results are sorted by total score, descending.

## License

MIT
