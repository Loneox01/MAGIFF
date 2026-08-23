# Client command reference

This directory adapts MAGIFF's services to external clients. Keep client
formatting and transport behavior here; deterministic query and resolution
logic belongs in `services/`.

## Discord commands

Discord slash-command parameters are separate form fields presented by Discord.
Users do not need to remember comma-separated text or manually parse a command
string.

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

## Adding another Discord command

1. Define its typed service behavior and failure outcomes in `services/`.
2. Add parsing, display formatting, and delivery under `integrations/`.
3. Add the Discord option schema to `jobs/register_discord_commands.py`.
4. Dispatch the command in `api/app.py` and add API/integration tests.
5. Deploy the API, then rerun `python -m jobs.register_discord_commands`.
