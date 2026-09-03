# Local data

Generated datasets in this directory are ignored by Git. The processed output is organized by how frequently each category changes:

Unstructured report snapshots use one Markdown file per source document so
publication dates, entities, and source URLs remain attributable:

```text
raw/
└── reports/
    ├── YYYY-MM-DD/
    │   ├── manifest.json
    │   └── source_name/
    │       └── YYYY-MM-DD_article-slug.md
    └── sources/
        └── fantasypros/
            ├── items/
            ├── versions/
            └── latest_run.json
```

The initial seed records use `content_mode: source_summary`: they contain
source-grounded summaries and metadata rather than copied article bodies.
Owned or licensed full-text sources can use the same frontmatter structure
with a different content mode later.

```text
processed/
├── manifest.json
├── orchestration/
│   └── request_router.sqlite3
├── reports/
│   ├── local_rag.sqlite3
│   ├── latest_run.json
│   ├── load_latest_run.json
│   ├── evaluation_logs/
│   └── documents/
│       └── fantasypros/
│           └── {external_report_id}.json
├── reference/
│   ├── players.parquet
│   ├── player_external_ids.parquet
│   └── teams.parquet
├── current/
│   ├── player_status.parquet
│   ├── player_ecr.parquet
│   ├── games.parquet
│   └── depth_chart_entries.parquet
└── seasons/
    ├── 2024/
    │   ├── games.parquet
    │   ├── player_weekly_stats.parquet
    │   ├── player_season_stats.parquet
    │   ├── player_weekly_rosters.parquet
    │   ├── player_snap_counts.parquet
    │   ├── team_weekly_stats.parquet
    │   ├── player_ecr.parquet
    │   └── depth_chart_entries.parquet
    └── 2025/
        ├── games.parquet
        ├── player_weekly_stats.parquet
        ├── player_season_stats.parquet
        ├── player_weekly_rosters.parquet
        ├── player_snap_counts.parquet
        ├── team_weekly_stats.parquet
        ├── player_ecr.parquet
        └── depth_chart_entries.parquet
```

- `reference/` contains slowly changing player and team identity data.
- `current/` contains overwriteable active-season and player-status data.
- `seasons/` contains normalized historical tables partitioned by season.
- `reports/` contains database-ready normalized report documents, the latest
  metadata-processing run summary, and the rebuildable local keyword/vector
  retrieval index. Its ignored `evaluation_logs/` directory contains local
  benchmark output.
- `orchestration/` contains the rebuildable request-route cache and telemetry.
- `manifest.json` records processing times, output shapes, and unmatched rows.
