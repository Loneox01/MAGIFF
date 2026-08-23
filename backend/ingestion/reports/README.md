# Report ingestion

FantasyPros NFL news can be fetched either into the local replayable snapshot or
through the deployable continuous refresh job. Both paths use the same provider
payload hash and normalized document contract.

Set the key in the project-root `.env`:

```text
FANTASYPROS_API_KEY=...
```

## Local source-only ingestion

Run from `backend/` to inspect/preserve a provider response without processing
or uploading it:

```bash
python -m ingestion.reports.fantasypros --limit 20
```

Raw items are stored idempotently under
`data/raw/reports/sources/fantasypros/`. Repeated identical items are skipped;
changed items retain content-addressed versions. `latest_run.json` reports the
inserted, updated, unchanged, and failed counts.

For offline development, pass a saved API response with `--fixture path.json`.

## Continuous Supabase refresh

After applying `20260821235900_create_report_ingestion_runs.sql`, run:

```bash
python -m jobs.refresh_reports \
  --limit 20 \
  --daily-request-budget 40 \
  --trigger manual
```

One job invocation makes **at most one FantasyPros request** and asks that call
for **20 reports**. It then:

1. reserves one request in Supabase's UTC-day quota ledger and obtains a
   30-minute overlap lease;
2. compares provider payload hashes with current `reports` rows;
3. skips metadata extraction and embeddings for unchanged reports;
4. resolves player identities against Supabase rather than ignored Parquet
   files;
5. processes new/changed reports in a temporary directory and transactionally
   loads their document, version, player links, and chunk; and
6. records counts, model tokens, embedding work, publication coverage, status,
   and any error in `report_ingestion_runs`.

The GitHub Actions workflow runs at minute 17 of every hour: normally **24 API
requests/day**, comfortably below both the application's **40 per rolling 24
hours budget** and
the provider's stated **50/day limit**. Manual workflow runs share the same
database budget, so they cannot silently exceed it. Provider requests are not
automatically retried because each attempt may consume quota.

The two feed-window flags help tune the per-call report count later:

- `feed_window_saturated=true` means FantasyPros returned all 20 requested rows.
- `possible_coverage_gap=true` means the window was full and every valid row was
  new or changed; more unseen reports may exist beyond that single page.

Required GitHub repository secrets are `FANTASYPROS_API_KEY`, `OPENAI_API_KEY`,
`SUPABASE_URL`, and `SUPABASE_SECRET_KEY`. The secret Supabase key is used only
inside the backend job and must never be exposed to Vercel/browser code.

## One-time historical backfill

The news endpoint has no documented page cursor. The backfill therefore uses
its supported FantasyPros-player-ID filter and builds a relevance-first queue
from current ECR, prior-season PPR production, and active QB/RB/WR/TE records.
It uses the same hashes, metadata processor, embedding loader, ingestion lease,
and rolling request ledger as the continuous job.

First preview the next chunk. This reads Supabase state but does not call
FantasyPros or OpenAI:

```bash
python -m jobs.backfill_reports \
  --from 2026-01-01 \
  --target-reports 200 \
  --max-requests 2 \
  --plan
```

For the first live test, make one provider request:

```bash
python -m jobs.backfill_reports \
  --from 2026-01-01 \
  --target-reports 200 \
  --max-requests 1
```

If that output looks right, subsequent invocations can use the default bounded
two-request chunk:

```bash
python -m jobs.backfill_reports \
  --from 2026-01-01 \
  --target-reports 200 \
  --max-requests 2
```

Successful player feeds are recorded in `report_ingestion_runs.metadata` and
skipped on later invocations. Partial feeds remain eligible for retry. The
target counts only reports newly added by this backfill, and database report
IDs still deduplicate anything the hourly feed already captured. A manual run
automatically pauses when the shared 40-request rolling budget or ingestion
lease prevents another call.

Each player request asks for up to 100 reports and then applies the inclusive
publication cutoff locally. If `possible_coverage_gap=true`, the provider filled
that player's entire exposed window, so older news may remain inaccessible
without a documented cursor. This is telemetry, not permission to make an
unbounded retry loop.
