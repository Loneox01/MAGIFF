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
    `--> search_reports
             |
             `--> report planner + Supabase entity grounding
                  + metadata filters + hybrid retrieval + reranking
    |
    v
main answer agent (Terra)
```

Structured domains currently map to player lookup, player statistics, team
statistics, schedules, rosters/depth charts, and ECR. Multiple domains and the
report capability can be selected for mixed questions. If the router call fails,
the terminal logs the failure and the main agent receives all registered
capabilities as a safe availability fallback.

Routes and token telemetry are stored in the ignored
`data/processed/orchestration/request_router.sqlite3` file. Identical requests
reuse the route for the current date. Configure the router model with
`OPENAI_ROUTER_MODEL`; the default is `gpt-5.6-luna`.

Web search is deliberately absent from the route enum and active tool registry.
Its commented definition remains in `main.py`. Enabling it later requires a web
route plus a local-evidence-first fallback policy; it should not mask ambiguous
identity, invalid planning, or ordinary tool errors.
