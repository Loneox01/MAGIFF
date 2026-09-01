# Waiver advisor

This package is a read-only laboratory for in-season waiver decisions. It does
not submit claims, add players, drop players, or mutate a Sleeper league.

The default model packet includes the managed roster, league/waiver settings,
current matchup and transactions, Sleeper trends, and a small group of top
available FantasyCalc candidates. The complete loaded market pool stays behind
bounded discovery tools:

- `rank_available_players`: top market value, trend, or ECR with optional
  position/team filters.
- `get_available_player`: named-player availability plus prior-season and
  current depth-chart context.
- `get_recent_news`: deterministic newest-first maintained news.
- `search_reports`: optional deeper contextual report retrieval.

The model first returns a typed preliminary shortlist. Code then retrieves the
latest maintained news for every proposed add and drop. A separate final pass
must select only from those news-checked names and returns enumerated actions,
roles, priorities, and time horizons. `no_action` is a supported result.

FantasyCalc's endpoint is public but undocumented. Its data is treated as a
best-effort market signal and failures must not be interpreted as player value.
No web-search fallback or waiver execution is enabled in this first test path.

From `backend/`:

```bash
python -m waivers.cli --context-only
python -m waivers.cli "Review my waiver options for this week."
python -m waivers.cli "Do I have any meaningful RB upgrades available?" --json
```
