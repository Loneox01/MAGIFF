# Client command reference

This directory adapts MAGIFF's services to external clients. Keep client
formatting and transport behavior here; deterministic query and resolution
logic belongs in `services/`.

## Discord commands

Discord slash-command parameters are separate form fields presented by Discord.
Users do not need to remember comma-separated text or manually parse a command
string.

The test guild receives all commands. The UAI friends guild receives the stable
`/news` and `/stats` subset; its runtime access can be toggled independently
with `DISCORD_UAI_ENABLED`. Roster roulette remains test-only while its rules
and scoring are calibrated.

### `/ask`

Runs the routed MAGIFF agent and can use structured NFL tools and report
retrieval.

```text
/ask question:Who should I start at flex in Week 1?
```

### `/news`

Reads recent reports directly from Supabase, newest first. It does not invoke a
planner, embeddings, reranker, or answer model.

| Option | Required | Accepted value | Default |
|---|---:|---|---|
| `player` | No | Full or partial player name | Any player |
| `team` | No | Team abbreviation or official name | Any team |
| `count` | No | Integer from 1 through 10 | 3 |
| `detail` | No | `headlines`, `summary`, or `full` | `headlines` |
| `previews` | No | `true` or `false` | `false` |

Examples:

```text
/news
/news count:10 detail:headlines
/news player:Kenny Gainwell count:3
/news team:TB detail:summary
/news player:Kenny Gainwell team:TB count:3 detail:full
/news player:Michael Penix Jr. previews:true
```

Bare `/news` therefore stays compact: it returns only the three newest linked
headlines and their source/date metadata. Headlines remain clickable while
Discord's large link-preview cards are suppressed. Set `previews:true` to show
those cards.

`full` means the complete text stored by MAGIFF, which may be a provider news
blurb rather than the full article at the linked source. Full view is capped at
three reports to stay within Discord's message limits; use `summary` or
`headlines` for a wider list.

Player and team filters are resolved against the database before report
retrieval. The command never guesses among duplicate player names. A missing or
ambiguous filter returns matching candidates or a suggested retry, and no
report query is made until the identity is safe. If valid filters have no stored
news, remove one filter or run unfiltered `/news` to inspect current coverage.

### `/stats`

Reads structured NFL data directly from Supabase without invoking an LLM. Its
subcommands cover one player, player leaderboards, one team, team leaderboards,
and field discovery.

The `formula` and `minimum_field` inputs use Discord's native autocomplete,
backed by the same field catalogs as processing and the agent tools. Select a
field, type an operator, then select another field—for example:

```text
/stats player player:A.J. Brown season:2025 view:receiving
/stats player player:A.J. Brown season:2025 formula:receiving_yards / targets
/stats player player:A.J. Brown season:2025 week:1 formula:fantasy_points_ppr
/stats leaders formula:receiving_yards / targets position:WR minimum_field:targets minimum_value:50
/stats team team:PHI season:2025 perspective:defense view:summary
/stats team-leaders formula:points_allowed / games perspective:defense
/stats fields scope:player-season search:receiving
```

Only catalog fields, numbers, `+`, `-`, `*`, `/`, and parentheses are accepted.
Autocomplete helps assemble an expression, but the backend always validates the
submitted formula because Discord permits arbitrary text. Omitting `season`
selects the latest stored season available for that query. Player and team
names use the same conservative resolution and ambiguity punts as `/news`.

### `/game roster`

Starts a seven-pick roster-roulette game without invoking an LLM. Each roll
selects an unused NFL team and one open roster slot. The season's highest PPR
scorer for that team and position is attached to the roll. FLEX resolves to an
RB, WR, or TE.

```text
/game roster
/game roster season:2024
/game roster season:2025 reveal:true
```

| Option | Required | Accepted value | Default |
|---|---:|---|---|
| `season` | No | A stored completed NFL season | Latest stored season |
| `reveal` | No | `true` or `false` | `false` |

The lineup is QB, two RBs, two WRs, one TE, and one FLEX. Accepted teams and
players cannot repeat. Every game has one team reroll and one position reroll;
the corresponding button is disabled after use. Hidden mode shows only the
rolled team and position until the final reveal, while `reveal:true` also shows
the player and season PPR total during each roll. The current team logo appears
in the roll card.

The final total maps to a 0–17 through 17–0 record using padded score anchors:
800, 850, 900, 100-point middle steps through 2,200, then 2,250 and 2,300. The
closest anchor determines the record, with exact ties favoring the lower one.
Sessions, picks, users, and actions persist in Supabase; versioned buttons and
database constraints reject stale clicks, duplicate deliveries, and repeated
teams or players.

## Adding another Discord command

1. Define its typed service behavior and failure outcomes in `services/`.
2. Add parsing, display formatting, and delivery under `integrations/`.
3. Add the Discord option schema to `jobs/register_discord_commands.py`.
4. Dispatch the command in `api/app.py` and add API/integration tests.
5. Deploy the API, then rerun `python -m jobs.register_discord_commands`.
