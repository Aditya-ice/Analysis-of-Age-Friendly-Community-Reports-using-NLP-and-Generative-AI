# Local ingestion and recovery

Use Python 3.12, `uv sync --frozen --group dev`, PostgreSQL with pgvector, and local Tesseract with English language data. The API never runs extraction or hosts PDFs. Keep the administrator database URL and Google key in an untracked `.env` or process environment.

Before any Google call, verify the project has billing disabled and check its actual quota in Google AI Studio. Set `ELDERHELP_GOOGLE_API_KEY`, then explicitly set `ELDERHELP_FREE_TIER_CONFIRMED=true`. The application cannot inspect your billing state. `ELDERHELP_EMBEDDING_DAILY_LIMIT=200` is an admission ceiling, not a promise of account quota; lower it when required. No Vertex, paid ranking or Document AI calls occur in the v2 ingestion path.

```sh
uv run elderhelp manifest validate
uv run elderhelp corpus reconcile --dry-run
uv run elderhelp corpus reconcile --apply
uv run elderhelp ingest
# Save the returned generation UUID, including after partial failure:
uv run elderhelp ingest resume GENERATION_UUID
uv run elderhelp corpus validate GENERATION_UUID
uv run elderhelp corpus activate GENERATION_UUID
```

The manifest pins the two existing files by SHA-256. Changed bytes require an explicit checksum approval and a new staged revision. HTTPS downloads require exact approved hostnames, revalidate redirects, reject non-public addresses, and pin connections to the validated IP while checking TLS against the original hostname. Downloads and local files have byte/page limits. Local paths must resolve under `ELDERHELP_SEED_ROOT`.

Extraction retains normalized page text, exact span offsets, bounding boxes, printed labels, heading sections and extraction confidence. It uses a column-order heuristic, repeated margin suppression, table row/header pairing, and selective local Tesseract OCR. These are heuristics requiring corpus evaluation, not claims of perfect layout understanding. Uncertain table fragments are excluded from evidence; unreadable pages quarantine the report. Uniform blank pages are preserved as blank provenance and excluded from search.

Chunks use locked `tiktoken`/`cl100k_base`, target 350 tokens with an overlap of up to 50, and stable IDs from source boundaries and configuration. Paragraphs/lists remain together when possible. Long blocks are split into exact smaller spans. Embedding 2 receives **one text input per call**, with compatible document/query prefixes, at most two concurrent calls, one-vector/768-finite-value checks and L2 normalization. Cached successful vectors survive failures. See [Google embedding guidance](https://ai.google.dev/gemini-api/docs/embeddings).

Leases are renewed during extraction and checked before committing chunks. Quota exhaustion pauses the generation for explicit resume; ambiguous dispatched calls remain counted. Independent document failures are quarantined while other reports continue. Validated generations are immutable through the CLI. Database storage is checked before ingestion and estimated before indexing; 70% warns and 80% blocks new ingestion. For large corpora, review actual storage growth before raising limits.

The initial local audit produced 182 NYC chunks across 92 pages and 60 community-engagement chunks across 16 pages. No live embeddings or quality scores are claimed by this audit.

## Private backup and restore drill

Install PostgreSQL client tools at least as new as the server. Keep archives outside source control. The commands create private files and refuse to overwrite an existing archive. Supply passwords through configuration, never command-line URLs in committed scripts.

```sh
uv run elderhelp backup .local/backup-YYYY-MM-DD.dump
# Create a separate EMPTY database named elderhelp_restore_test using your administrator tools.
# Point ELDERHELP_DATABASE_URL at that disposable target, then:
uv run elderhelp restore-check .local/backup-YYYY-MM-DD.dump
uv run elderhelp corpus status
```

Restore is transactional, does not drop existing objects, and requires a database ending in `_restore_test`. Local verification restored the test schema, staged generations and counters with PostgreSQL 16/pgvector. A production handover must repeat the drill using the actual hosted corpus and verify an active generation plus search. Store backup metadata before pruning inactive generations; retain the active and previous compatible generations when storage allows.
