# Local data

Generated datasets in this directory are ignored by Git. The processed output is organized by how frequently each category changes:

```text
processed/
├── manifest.json
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
- `manifest.json` records processing times, output shapes, and unmatched rows.
