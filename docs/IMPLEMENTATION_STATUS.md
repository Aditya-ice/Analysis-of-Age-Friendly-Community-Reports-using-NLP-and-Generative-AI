# Reliable RAG implementation checkpoints

Baseline: `8ae26b8`. Original code/history, seed reports and mobile apps are preserved.

| Step | State | Verification |
|---|---|---|
| 1. Audit and v2 contract | Implemented | Contract/schema tests; source audit and reproduced grounding failures |
| 2. Corpus lifecycle | Implemented | PostgreSQL 16/pgvector migrations; 27 tests, including metadata-only changes, duplicate hashes, exact-span validation, atomic activation, incompatible configuration rejection and withdrawal-safe rollback |
| 3. Extraction and ingestion | Pending | |
| 4. Retrieval | Pending | |
| 5. Verified answers/API | Pending | |
| 6. Evaluation | Pending | |
| 7. Browser/mobile clients | Pending | |
| 8. Free deployment/handover | Pending | |

Each completed step is committed and pushed separately. No quality scores, live-model results or deployment are claimed until actually measured. Human gold-label review, free-tier account configuration and live quota are external gates; paid services are prohibited.
