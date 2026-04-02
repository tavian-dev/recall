# recall Architecture Decision Record

## Overview
Hybrid search tool for markdown memory files. BM25F + ChromaDB semantic embeddings fused with Reciprocal Rank Fusion.

## Key Decisions

### BM25F over naive per-field BM25 (2026-04-02)
Switched from scoring body and title independently to BM25F: normalize TF per field, combine weighted TFs, apply saturation once. Fixes double-counting of saturation. Uses max(df) across fields for unified IDF.

### ChromaDB for semantic search (2026-04-02)
Chose ChromaDB (built-in sentence-transformer, SQLite backend) over Ollama or API-based embeddings. Zero external services. Graceful degradation to BM25-only if chromadb not installed.

### RRF over learned fusion (2026-04-02)
Using Reciprocal Rank Fusion to combine BM25 and semantic results. Simple, parameter-free, works well. Upgrade path: Bayesian BM25 (BB25) for principled probabilistic fusion.

## Future Direction
- Static embeddings (397x faster, 87% quality) when corpus grows
- BB25 for probabilistic fusion
- Relationship tracking between memory entries (supersedes, contradicts)
