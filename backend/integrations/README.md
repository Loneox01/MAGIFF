# Client command reference

This directory adapts MAGIFF's services to external clients. Keep client
formatting and transport behavior here; deterministic query and resolution
logic belongs in `services/`.

`sleeper.py` contains public, read-only draft and in-season league adapters.
The league adapter retrieves members, rosters, matchups, transactions, NFL
state, and trending adds/drops for the deterministic league context. Neither
adapter authenticates a Sleeper user or mutates a roster.

`fantasycalc.py` is a best-effort, read-only adapter for current FantasyCalc
redraft market values. The waiver advisor configures the request from the live
league's team count, quarterback format, and PPR scoring, then joins players by
Sleeper ID. FantasyCalc's endpoint is public but undocumented, so failures are
surfaced as missing market evidence rather than interpreted as zero value.

## Discord commands

Discord slash-command parameters are separate form fields presented by Discord.
Users do not need to remember comma-separated text or manually parse a command
string.

The test guild receives all commands. The UAI friends guild receives `/news`,
`/stats`, and the 17-0 Challenge; its runtime access can be toggled independently
with `DISCORD_UAI_ENABLED`. `/ask` remains test-only.

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

### `/game challenge`

Starts the seven-pick 17-0 Challenge without invoking an LLM. Each roll
selects an unused NFL team and one open roster slot. The season's highest PPR
scorer for that team and position is attached to the roll. FLEX resolves to an
RB, WR, or TE only after that position's ordinary roster slot or slots are full.

```text
/game challenge
/game challenge season:2024
/game challenge season:2025 reveal:true
/game challenge season:2025 scoring:ppg
```

| Option | Required | Accepted value | Default |
|---|---:|---|---|
| `season` | No | A stored completed NFL season | Latest stored season |
| `scoring` | No | `season_total` or `ppg` | `season_total` |
| `reveal` | No | `true` or `false` | `false` |

The lineup is QB, two RBs, two WRs, one TE, and one FLEX. Accepted teams and
players cannot repeat. Every game has one team reroll and one position reroll;
the corresponding button is disabled after use. The red forfeit button ends
the run and reveals its saved picks. Hidden mode shows only the
rolled team and position until the final reveal, while `reveal:true` also shows
the player and selected score during each roll. The current team logo appears
in the roll card.

Season-total mode selects each team-position's highest full-season PPR scorer.
PPG mode instead selects its highest PPR average across that player's stored
stat-row games for the team. The default remains season totals.

The final score maps to a 0–17 through 17–0 record using a dynamic scale for the
selected season and scoring mode. The service calculates the lowest and highest
fully legal rosters, places the 0–17 boundary at 10% of that attainable range and
the 17–0 boundary at 85%, and evenly spaces records 1–16 between them. Season-total
boundaries are rounded to the nearest 25 points and PPG boundaries to the nearest
whole point.
Sessions, picks, users, and actions persist in Supabase; versioned buttons and
database constraints reject stale clicks, duplicate deliveries, and repeated
teams or players.

## Adding another Discord command

1. Define its typed service behavior and failure outcomes in `services/`.
2. Add parsing, display formatting, and delivery under `integrations/`.
3. Add the Discord option schema to `jobs/register_discord_commands.py`.
4. Dispatch the command in `api/app.py` and add API/integration tests.
5. Deploy the API, then rerun `python -m jobs.register_discord_commands`.
