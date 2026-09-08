# Retrieval checkpoint

The research retriever is independent of generation. It searches only currently approved reports in the active generation. Dense retrieval requires the exact configured embedding/extraction/chunking fingerprint. Keyword fallback can continue using an older active index during an embedding-configuration mismatch.

Simple standalone questions use no planning call. Follow-ups/comparisons use one structured planning call, producing a resolved question and up to three component queries. User filters remain authoritative. Both reranking and the subsequent generator receive the resolved question. Unresolved references require clarification.

Each component retrieves up to 30 dense and 30 PostgreSQL keyword candidates. Keyword ranking weights titles/sections above body text; current title metadata is searchable even without re-embedding. Exact cosine search is used intentionally; there is no approximate index on v2 vectors. Benchmark filtered HNSW against exact search at 10,000 active chunks before introducing it, as [pgvector's filtering guidance](https://github.com/pgvector/pgvector#filtering) explains the recall implications.

Reciprocal Rank Fusion (`k=60`) merges and deduplicates near-overlapping spans. At least one candidate per component survives the 20-candidate shortlist. Evidence selection applies a soft report-diversity preference, at most eight blocks, and a 6,000-token serialized evidence budget. Child spans always survive parent expansion; if the budget cannot accommodate an additional complete child, the block is omitted. Verification must still check actual question coverage: retrieval membership alone is not proof of support.

## Local reranker

```sh
uv sync --frozen --group dev
uv run python tools/download_reranker.py
```

The [official MiniLM cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/main/onnx) is pinned by revision and SHA-256 in `contracts/reranker-manifest.json`. The build/admin download chooses quantized ARM64 or AVX2 weights, approximately 23 MB, plus the tokenizer. Runtime loads local verified artifacts using CPU ONNX Runtime, one inference at a time, two-window batches and one native inference thread. No PyTorch or remote ranking endpoint is required. Candidates are scored across overlapping windows; oversize queries/candidates fail explicitly into a recorded RRF fallback instead of silently losing their ending.

Unit/integration checks cover PostgreSQL lexical/vector retrieval and metadata filters, current withdrawal, resolved follow-ups, child preservation, component selection, fallback, and actual ONNX scoring on short/long text. These tests establish engineering behavior. Retrieval quality on the real seed corpus is measured separately in step 6; no Recall@8 score is claimed yet.
