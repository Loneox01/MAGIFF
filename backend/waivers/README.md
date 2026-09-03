# Waiver advisor

This package is a read-only laboratory for in-season waiver decisions. It does
not submit claims, add players, drop players, or mutate a Sleeper league.

The default model packet includes the managed roster, league/waiver settings,
current matchup and transactions, Sleeper trends, league-scoring-adjusted
weekly projections, and a small group of top available FantasyCalc candidates.
The complete loaded market and projection pools stay behind bounded discovery
tools:

- `rank_available_players`: top market value, trend, ECR, or current-week
  projection with optional position/team filters.
- `get_available_player`: named-player availability plus prior-season and
  current depth-chart context.
- `get_player_week_outlook`: one managed or available player's projection,
  opponent, and date for a selected week.
- `rank_streaming_defenses`: compare the current D/ST with available defenses
  over a one- to three-week horizon.
- `get_recent_news`: deterministic newest-first maintained news.
- `search_reports`: optional deeper contextual report retrieval.

The model first returns a typed preliminary shortlist. Code then retrieves the
latest maintained news for every proposed add and drop. A separate final pass
must select only from those news-checked names and returns enumerated actions,
roles, priorities, and time horizons. `no_action` is a supported result.

FantasyCalc's endpoint is public but undocumented. Its data is treated as a
best-effort market signal and failures must not be interpreted as player value.
Sleeper's projection feed is also undocumented and isolated behind a best-effort
read-only adapter. Point estimates are calculated from the feed's projected stat
components using the league's actual Sleeper scoring settings; neither the raw
feed nor the calculated result guarantees health, role, or outcomes. A feed
failure leaves the rest of the waiver context usable.

No web-search fallback or waiver execution is enabled in this test path.

From `backend/`:

```bash
python -m waivers.cli --context-only
python -m waivers.cli "Review my waiver options for this week."
python -m waivers.cli "Do I have any meaningful RB upgrades available?" --json
```
