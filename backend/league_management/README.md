# Read-only league management context

This package is the deterministic boundary between a live Sleeper league and
future MAGIFF waiver, lineup, trade, and news-response policies. It does not
invoke an LLM and cannot modify a Sleeper roster.

The builder concurrently reads the public Sleeper user, league, members,
rosters, NFL state, current matchup, current-week transactions, and global
trending adds/drops. It verifies roster ownership by immutable Sleeper user ID,
enriches player IDs from Supabase, and removes every rostered Sleeper ID from a
compact current-ECR candidate list.

Run a live read from `backend/`:

```bash
python -m league_management.cli \
  --league-id YOUR_SLEEPER_LEAGUE_ID \
  --user YOUR_SLEEPER_USERNAME
```

Use `--json` for the complete provider-neutral snapshot or `--agent-json` for
the compact payload intended for later policy-specific agents. The CLI also
accepts `SLEEPER_LEAGUE_ID` and `SLEEPER_USERNAME` (or `SLEEPER_USER_ID`) from
the project-root `.env`; these are public identifiers, not credentials.

The initial market list is deliberately labeled: it is the highest current ECR
players absent from all league rosters, not a complete waiver wire, projection,
or recommendation. Sleeper's documented public API is read-only and does not
provide its UI projections. Authenticated UI observation and action execution
remain separate future layers.

The policy-specific read-only waiver workflow lives separately under
`waivers/`. It consumes this league snapshot but keeps the broader free-agent
and FantasyCalc pools behind bounded tools instead of serializing them into the
default model prompt.

The policy-specific read-only lineup workflow lives under `lineups/`. It skips
the unrelated ECR market query, joins the managed roster to Sleeper's
best-effort weekly projection and injury payload, and validates every proposed
starter against roster membership and legal slot eligibility.
