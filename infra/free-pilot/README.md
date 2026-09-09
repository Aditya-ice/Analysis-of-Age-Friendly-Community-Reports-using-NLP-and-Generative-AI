# Free pilot operations — deployment remains gated

This is a preparation bundle, not a deployed environment. Do not import the
Blueprint until the release checklist below passes: importing creates an initial
deployment even though `autoDeployTrigger: off` disables subsequent automatic
ones. Do not merge a branch just to deploy it. The nine original commits and
legacy Google Cloud configuration remain preserved.

## Operating envelope and account setup

Use one Render **Free** web service, a separate Supabase **Free** project, and a
Gemini Developer API project with billing disabled. Confirm actual model access
and the project's quota before enabling `ELDERHELP_FREE_TIER_CONFIRMED=true`.
Keep that flag false during setup; there is no paid fallback. A provider's public
price table does not establish an individual account's allowance.
The selected `gemini-3.6-flash` and `gemini-embedding-2` text inputs are currently
listed with standard free-tier access; no batch/paid mode is used.
[Google pricing](https://ai.google.dev/gemini-api/docs/pricing) (checked 2026-09-09).

Render currently supplies 512 MB / 0.1 CPU, sleeps after inactivity and has an
ephemeral filesystem. Leave payment methods absent and do not enable paid plans,
add-ons, previews or keep-alive automation. Read the current bandwidth/build
allowances before setup; exhaustion can suspend the pilot. No SSH or remote
migration job is assumed. [Render free limits](https://render.com/docs/free),
[Blueprint reference](https://render.com/docs/blueprint-spec).

Use Supavisor **session mode, port 5432**, for an IPv4 serving connection. Use the
direct connection for local administration when IPv6 is available. Copy the
project's server root certificate from its dashboard and use verified TLS;
never solve certificate errors by disabling verification. Confirm the free DB
allowance and inactivity policy in the account before ingestion.
[Supabase connections](https://supabase.com/docs/guides/database/connecting-to-postgres),
[Supabase pricing](https://supabase.com/pricing).

## Reproducible local preparation

```sh
uv python install 3.12
uv sync --frozen --group dev --extra ingestion
uv run python tools/download_reranker.py
uv run elderhelp manifest validate
```

Set local environment variables privately. `ELDERHELP_DATABASE_URL` uses the
`postgresql+asyncpg://` scheme and a URL-encoded password. Do not append `sslmode`
query parameters; the explicit TLS settings are shared by API/migrations/backups.
On macOS, if Python skips a hidden editable-install `.pth`, set
`PYTHONPATH=backend` for the commands below.

```sh
export ELDERHELP_DATABASE_TLS=verify-full
export ELDERHELP_DATABASE_CA_FILE=/absolute/private/path/supabase-ca.crt
uv run alembic upgrade head
```

Use migration-owner credentials only locally. Run `roles.sql` using `psql` as
that owner after migrations. Set psql's `PGHOST`, `PGPORT`, `PGUSER`, `PGDATABASE`,
`PGSSLMODE=verify-full`, and `PGSSLROOTCERT` privately; let psql prompt for the
password. The script is reapplicable and touches only named ElderHelp tables.

```sh
psql -v ON_ERROR_STOP=1 -f infra/free-pilot/roles.sql
```

Create separate login users in the private SQL console (no password in source):

```sql
CREATE ROLE elderhelp_api LOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE elderhelp_admin LOGIN NOSUPERUSER NOBYPASSRLS;
GRANT elderhelp_serving TO elderhelp_api;
GRANT elderhelp_ingestion TO elderhelp_admin;
```

Set passwords with `\password elderhelp_api` and `\password elderhelp_admin` in
psql. Do not grant ownership, superuser, BYPASSRLS, or ingestion membership to
the API user. Serving can read corpus/access state and update quota counters;
it cannot change reports or indexes. Keep migration-owner credentials for schema
migrations and full backups. Disable the Supabase Data API in dashboard settings
for this dedicated project; no client uses Supabase public keys. Reapply and test
role policy whenever a migration adds tables. Do not grant public schema CREATE
to the API login.

Use the local administrator connection for ingestion and corpus commands:

```sh
uv run elderhelp corpus reconcile --dry-run
uv run elderhelp corpus reconcile --apply
uv run elderhelp ingest
uv run elderhelp ingest resume GENERATION_UUID
uv run elderhelp corpus validate GENERATION_UUID
uv run elderhelp corpus activate GENERATION_UUID
uv run elderhelp corpus status
```

The `ingest` command needs a separately verified free Google allowance. Wait and
resume on quota exhaustion. Keep PDFs and extracted files local; upload only
approved index data to PostgreSQL. Never put PDFs in the serving image or Render
filesystem. Warn at 70% database allowance; ingestion stops at 80%.

## Backup, restore and pruning

Install PostgreSQL client tools matching the server major version. If several
versions are installed, set `ELDERHELP_POSTGRES_BIN_DIRECTORY` explicitly (for
example `/opt/homebrew/opt/postgresql@16/bin` on macOS). A version-14 dump client
cannot back up a version-16 server.

Keep encrypted/offline copies of administrator-only backups. Backups contain
report text, invite hashes and quota counters; do not commit them. Before schema
changes, corpus activation, or pruning, use migration-owner credentials to make
a full backup (an RLS-restricted role is not a complete backup identity).

```sh
uv run elderhelp backup .local/backups/before-change.dump
```

Create a fresh disposable local database whose name ends in `_restore_test`.
Point `ELDERHELP_DATABASE_URL` at it and use `ELDERHELP_DATABASE_TLS=disable` only
for that local connection. Then run:

```sh
uv run elderhelp restore-check .local/backups/before-change.dump
uv run elderhelp corpus status
uv run elderhelp corpus validate RESTORED_ACTIVE_GENERATION_UUID
```

Verify active/previous IDs, approved report counts and exact-span validation
against the original. A restore drill on one database is not a hosted disaster
recovery guarantee. The restore command refuses a non-test database name and
never drops existing tables. A failed restore rolls back its transaction.

Re-select the intended administrator connection before maintenance:

```sh
uv run elderhelp corpus prune OLD_GENERATION_UUID .local/generation-archives
uv run elderhelp corpus prune OLD_GENERATION_UUID .local/generation-archives --apply
uv run elderhelp corpus rollback
```

Pruning protects active/previous generations, rejects unfinished jobs, archives
rebuild metadata before deletion, and removes only no-longer-referenced search
chunks/embeddings. It retains immutable revisions and page/span provenance.
Staging generations and unassociated embedding cache entries are not reclaimed
by this command. Retain the full backup and local PDFs: the metadata archive
alone is not a data backup. PostgreSQL reuses deleted space; do not run disruptive
`VACUUM FULL` automatically. Never roll back report withdrawals.

## Release checklist (all required before import)

- Backend CI, PostgreSQL tests, browser tests, iOS simulator and Android emulator
  checks pass for the reviewed commit. No draft text appears before verification.
- The 120-case evaluation has actual human-reviewed labels and output review.
  Both development and held-out quality reports pass every gate. Run and record
  keyword/dense/hybrid/reranked ablations; do not substitute synthetic measurements.
- Record a restore drill on the intended schema and a Linux serving-memory run.
  The CI image test has a 512 MiB cap and requires peak RSS below 80%; its two
  synthetic reranking workloads are not a live-answer latency measurement.
- Complete and record local two-answer/ten-request admission tests. Record live
  warm verified-completion p95 (target 30 s), cold start separately, provider token
  use, and quota consumption. Do not weaken verification if the target fails.
- Confirm zero-billing/free-tier account settings, model availability, limits,
  rights/approval manifest and invite revocation. Set lower limits when required.
- Record a reviewed Git commit and configuration/index fingerprints. Missing
  evidence blocks release; a passing code test alone does not authorize deployment.

## Manual hosted setup after gates pass

1. In Render, import `infra/free-pilot/render.yaml` from a reviewed, merged commit.
   Confirm Free plan, one service, no databases/jobs/disks/previews. Automatic
   deploys remain off; choose each verified commit manually.
2. Add `supabase-ca.crt` as a Render secret file at `/etc/secrets/supabase-ca.crt`.
   Set the serving-role database URL and Google key as server secrets. Share the
   generated token secret privately with the local invite CLI; it must match the
   server. Never put token secrets/invite codes in browser or native builds.
3. Keep same-origin browser CORS empty. Add only exact HTTPS origins if needed.
   Defaults cap answers at 20/day, 6/minute and two concurrent workflows; lower
   them to the verified provider allowance. One worker, pool size two, no overflow.
4. After operator confirmation, enable the free-tier flag. Create an invite using
   `uv run elderhelp corpus invite-create`; securely distribute it only to testers.
   Revoke with `uv run elderhelp corpus invite-revoke INVITE_UUID`.
5. Check `/healthz`, `/readyz`, access rejection, library/search, then a curated
   non-sensitive question through the browser and both apps. Verify partial
   answers, citation dates, cancellation, history deletion, quota fallback and
   restarting without corpus loss. Record results before calling the pilot ready.

The serving image performs no ingestion or automatic migration. Readiness checks
local components and the active index without spending model quota. A missing
Google allowance disables answers while keyword search remains available.

## Troubleshooting and handover

| Symptom | Action |
|---|---|
| Host slow after idle | Wait for wake-up; distinguish cold from warm latency. Do not add keep-alive traffic. |
| 401 | Re-enter an unrevoked invite; saved history remains local. |
| 429 | Respect Retry-After; use keyword search; reduce account limits if necessary. |
| 503 / not ready | Check Supabase is unpaused, verified TLS, active compatible index, pinned reranker and token secret. Do not expose connection strings in logs. |
| Certificate error | Check hostname, CA file and dashboard certificate; never disable verification. |
| DB approaches allowance | Stop ingestion, back up/restore-check, preview inactive-generation pruning. |
| Verified quality fails | Keep deployment blocked; inspect curated traces and category failure analysis. |
| Provider unavailable | Search-only mode; no automatic paid model or infrastructure fallback. |

Ordinary logs contain request IDs, versions, counts, timings and error categories;
do not enable SQL echo, raw ASGI access logs, or provider prompt logging. The
explicit curated evaluation mode is the only place to retain research traces.
Existing `infra/terraform` is a future paid-cloud reference and is excluded from
these instructions. This pilot has no always-on availability guarantee.

## Recorded local/CI results

At `f301ec1`, [Linux memory evidence](results/linux-memory-f301ec1.json) records
307,359,744-byte peak RSS (293.1 MiB) under 512 MiB / 0.1 CPU limits; CI run
34315758422 passed. Two synthetic long-passage reranking workloads took 145.4 s.
This is a stress-path result, not real-answer p95; the live 30-second target is
unmeasured. Cancellation now stops remaining candidates/windows after the current
native batch, while retaining the inference lock until the worker exits.

The [local restore drill](results/restore-20260909.json) preserved active/previous
pointers, counts and exact spans for a synthetic fixture. A drill against the
actual hosted seed corpus is still required. No secrets or database dumps are
in these committed result files.

The repeat at `a9d9363` also passed: [final code memory evidence](results/linux-memory-a9d9363.json)
records 310,333,440 bytes (296.0 MiB) and 142.3 s for the same synthetic stress
workload (CI run 34375598071). These repeated measurements do not establish live
answer latency or corpus accuracy.
