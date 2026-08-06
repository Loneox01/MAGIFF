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

Depth-chart source formats are isolated in
`normalization/depth_charts.py`. Add another adapter there if nflverse changes
its schema for an older or future season.
