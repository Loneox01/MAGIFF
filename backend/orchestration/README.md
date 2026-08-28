# Agent orchestration

The request router runs once before the main agent. It uses `gpt-5.6-luna` by
default to choose a compact set of evidence capabilities and structured tool
domains; it does not answer the question or generate SQL.

```text
user request
    |
    v
request router (Luna, structured output, daily local cache)
    |
    +--> selected structured NFL tool groups
    |
    +--> search_reports
             |
             `--> report planner + Supabase entity grounding
                  + metadata filters + hybrid retrieval + reranking
    |
    `--> hosted web search (explicit/live route or weak-report fallback only)
    |
    v
main answer agent (Terra)
```

Single-player model tools accept either a canonical player name or an internal
UUID as `player_ref`. A request-scoped resolver translates names to UUIDs once,
returns explicit ambiguity candidates instead of guessing, and leaves the
repository/tool implementations UUID-based. Independent function calls emitted
in one model response execute concurrently in a bounded worker pool; their
outputs are returned in original call order, while genuinely dependent work
continues in a later model round.

Structured domains currently map to player lookup, player statistics, team
statistics, schedules, rosters/depth charts, and ECR. Multiple domains and the
report capability can be selected for mixed questions. If the router call fails,
the terminal logs the failure and the main agent receives all registered
capabilities as a safe availability fallback.

Routes and token telemetry are stored in the ignored
`data/processed/orchestration/request_router.sqlite3` file. Identical requests
reuse the route for the current date. Configure the router model with
`OPENAI_ROUTER_MODEL`; the default is `gpt-5.6-luna`.

Hosted web search is a controlled fallback. The router exposes it immediately
only for an explicit public-web request or a genuinely live information need.
Ordinary current questions use maintained reports first; a failed, weak, or
no-evidence report result deterministically enables one required web-search
round. Partial report evidence does not trigger the fallback. Web calls and
cited sources are returned in API telemetry, while hosted-tool fees remain
separate from the text-token cost estimate.
