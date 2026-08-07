# Fantasy-market ingestion

Raw files retain each provider's schema. Player matching, common columns, and
multi-source consensus belong in processing, not ingestion.

## Verified source status

| Source | Data | Current ingestion path | Why |
| --- | --- | --- | --- |
| nflverse | Current and historical FantasyPros ECR | Direct through `nflreadpy` | `load_ff_rankings()` and `load_ff_playerids()` are documented loaders. |
| Sleeper | ADP/rankings | Provider file import | The public API documents players, leagues, and drafts, but not a scoring-specific aggregate ADP feed. Draft-derived ADP can become its own adapter later. |
| Yahoo | ADP/rankings | Provider file import | The documented Fantasy Sports API uses OAuth and league/player resources; a global ADP resource has not been verified. |
| ESPN | ADP/rankings | Provider file import | No documented public ADP API has been verified. Do not depend on an undocumented endpoint. |
| FantasyPros | ADP | Provider file import | A dedicated API adapter can be added if API access and usage terms are selected. |

## Commands

Run these from `backend/`:

```bash
python -m ingestion.fantasy.nflverse player-ids --season 2026
python -m ingestion.fantasy.nflverse current-ecr --season 2026
python -m ingestion.fantasy.nflverse ecr-archive
python -m ingestion.fantasy.preview --season 2026
python -m ingestion.fantasy.preview --season 2026 --include-archive
```

Import a provider download without changing its columns:

```bash
python -m ingestion.fantasy.adp_file ~/Downloads/adp.csv \
  --provider sleeper \
  --season 2026 \
  --scoring ppr \
  --source-reference "Sleeper export acquired 2026-08-07"
```

Daily snapshots are written under:

```text
data/raw/fantasy/
├── nflverse/
│   ├── ecr/{season}/snapshots/{date}/
│   └── player_ids/{season}/snapshots/{date}/
├── espn/adp/{season}/snapshots/{date}/
├── sleeper/adp/{season}/snapshots/{date}/
└── yahoo/adp/{season}/snapshots/{date}/
```
