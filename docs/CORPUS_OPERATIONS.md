# Corpus lifecycle

The v2 migration is additive. Legacy reports/pages/chunks and the original Git history remain intact. Legacy vectors are never activated in a v2 index. PDF content hashes are not unique across report identities; duplicate documents are valid inputs.

Use administrator database credentials for migrations and these commands. Source/approval metadata is authoritative in `data/reports.yaml`. Updating metadata never requires an embedding call. Publication metadata is also snapshotted immutably at acquisition for historical attribution.

```sh
uv run alembic upgrade head
uv run elderhelp corpus reconcile --dry-run
# Review the changes above, then explicitly apply:
uv run elderhelp corpus reconcile --apply
uv run elderhelp corpus status
uv run elderhelp corpus validate GENERATION_UUID
uv run elderhelp corpus activate GENERATION_UUID
uv run elderhelp corpus rollback
```

Activation validates manifest coverage, completed jobs, embedding configuration, and exact source offsets. It changes one database pointer transactionally, with an advisory lock serializing competing activations. Rollback never restores historical approval: current withdrawals apply to every generation. Retrieval integration is delivered in step 4; this stage does not switch the serving API to the new engine.

Integration tests require a disposable database whose name ends in `_test`; the test fixture truncates only that database. CI runs migrations before the tests and supplies PostgreSQL 16 with pgvector. Local verification used PostgreSQL 16.15 and pgvector 0.8.6.

```sh
ELDERHELP_DATABASE_URL=postgresql+asyncpg://USER:PASS@localhost:5432/elderhelp_test uv run alembic upgrade head
ELDERHELP_TEST_DATABASE_URL=postgresql+asyncpg://USER:PASS@localhost:5432/elderhelp_test uv run pytest
```

Ingestion, storage-budget safeguards and backup commands follow in step 3. Neither this stage nor validation asserts human-reviewed retrieval quality.
