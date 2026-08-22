# NFL processing workflows

Processing is split by how often its source data changes:

- `reference.py` rebuilds player, external-ID, team, and baseline status tables.
- `current.py` overwrites available active-season tables and keeps only the
  newest depth-chart snapshot for each team.
- `historical.py` rebuilds one completed season and reduces timestamped depth
  charts to the final pregame snapshot for each team and NFL week.

Run these from the repository root as modules:

```bash
python -m backend.processing.workflows.reference
python -m backend.processing.workflows.current --season 2026
python -m backend.processing.workflows.historical --season 2024
python -m backend.processing.workflows.historical --season 2025
```

Fantasy-market processing uses the same internal player UUIDs while keeping
time-dependent ECR separate from player identity data:

```bash
python -m backend.processing.workflows.reference
python -m backend.processing.fantasy.ecr current --season 2026
python -m backend.processing.fantasy.ecr historical --season 2024 --reference-season 2026
python -m backend.processing.fantasy.ecr historical --season 2025 --reference-season 2026
```

The reference workflow adds selected FantasyPros, Sleeper, ESPN, Yahoo, CBS,
MFL, and PFR mappings to `player_external_ids.parquet`. Current ECR is written
to `current/player_ecr.parquet`; completed-season snapshots are written to
`seasons/{season}/player_ecr.parquet`. PPR redraft pages are labeled directly;
formats whose source page does not state a scoring system use
`scoring_format=source_default` rather than an unsupported assumption.

The Supabase loader can upload ECR without re-uploading the larger NFL tables:

```bash
python -m backend.database.load_supabase --workflow fantasy-current --season 2026
python -m backend.database.load_supabase --workflow fantasy-historical --season 2024
python -m backend.database.load_supabase --workflow fantasy-reference
```

Depth-chart source formats are isolated in
`normalization/depth_charts.py`. Add another adapter there if nflverse changes
its schema for an older or future season.

FantasyPros player-news reports are normalized separately from the seasonal NFL
tables. The provider player ID is matched directly to the existing external-ID
crosswalk; Luna supplies a bounded document type and material secondary-player
mentions, which are accepted only after deterministic player-catalog resolution:

```bash
cd backend
python -m processing.reports.fantasypros
```

Database-ready JSON documents are written to
`data/processed/reports/documents/fantasypros/`. Unchanged source content is
skipped when the raw hash, model, prompt version, and normalizer version match.
Use `--force` to deliberately rerun metadata extraction.

The scheduled `jobs.refresh_reports` workflow uses the same normalizer with a
bounded Supabase-backed catalog, so a deployed runner does not depend on ignored
local Parquet files. The local command above retains the Parquet catalog for
offline processing and replay.
