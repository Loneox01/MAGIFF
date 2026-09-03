# Lineup advisor

This package builds a read-only weekly lineup recommendation for one Sleeper
roster. It never submits a lineup change because the public Sleeper API does not
provide authenticated lineup mutation.

The verified context contains:

- Exact starter slots and current starter/bench/reserve placement.
- Legal slot eligibility derived from each player's position and the league's
  configured starter slots.
- Weekly Sleeper projections recalculated under the league's scoring settings.
- Player opponent, exact kickoff, projection timestamp, and live Sleeper injury
  designation, body part, notes, and injury-news timestamp when available.
- Code-enforced player locks derived from the stored NFL schedule.
- A deterministic legal projection-only baseline and the opponent's current
  projected lineup total.

The first model stage may retrieve maintained news or deeper reports, then
returns a preliminary legal lineup. Code automatically checks recent news for
every changed player, every health-designated roster player, and close
alternatives selected by the model. A separate final stage returns typed advice,
which is rejected unless every slot, player, identity, and eligibility rule is
valid.

Sleeper's projection feed is undocumented and best-effort. If it is unavailable,
the roster still loads but projections and live injury fields may be missing.

## Automatic deadline review

`jobs.review_lineup` groups every starter and bench player at the next exact
kickoff into one slate. The scheduled job becomes eligible at T-75 minutes,
claims one idempotent Supabase review, runs the lineup advisor once, and sends
exactly one message for the whole slate to `magiff-log`. A no-change review is
logged without a ping. `CHANGE RECOMMENDED`, `REVIEW FAILED`, and
`EMERGENCY UPDATE` mention the configured owner.

After the first scheduled review, a changed Sleeper health/status snapshot
before kickoff creates one emergency review for that new snapshot. Locked
starters stay in their exact slots, locked bench players cannot enter, and
recommendations affecting only later kickoff windows are labeled provisional.

The persistent audit table is created by
`20260902120000_create_lineup_review_runs.sql`. Configure these GitHub Actions
secrets for `.github/workflows/review-sleeper-lineup.yml`:

```text
OPENAI_API_KEY
SUPABASE_URL
SUPABASE_SECRET_KEY
SLEEPER_LEAGUE_ID
SLEEPER_USERNAME
DISCORD_BOT_TOKEN
DISCORD_LINEUP_CHANNEL_ID
DISCORD_OWNER_USER_ID
```

`DISCORD_LINEUP_CHANNEL_ID` is the numeric ID of `magiff-log`, and
`DISCORD_OWNER_USER_ID` is the numeric user ID to mention. The bot needs View
Channel and Send Messages permission in that channel.

From `backend/`:

```bash
python -m lineups.cli --context-only
python -m lineups.cli "Set my best Week 1 lineup and explain the close calls."
python -m lineups.cli "Should I start my questionable receiver?" --json
python -m jobs.review_lineup --plan
python -m jobs.review_lineup --e2e-next
```

`--plan` is read-only and free: it prints every upcoming kickoff slate and its
review time. `--e2e-next` deliberately runs, stores, and posts one complete test
for the next upcoming slate even before T-75. Use `--dry-run --e2e-next` to run
the model without writing to Supabase or Discord.
