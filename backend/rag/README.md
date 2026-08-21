# Local report retrieval

This is the first, deliberately small RAG layer. It parses the report Markdown
records, treats each short report as one chunk, indexes text with SQLite FTS5,
and optionally caches OpenAI embeddings for local cosine search.

```text
rag/
├── planning/    # plans, grounding, typed lookups, identity routing
├── retrieval/   # index, embeddings, plan execution, reranking
├── evaluation/  # regression cases and runners
├── pipeline.py  # compact end-to-end search boundary for the agent
├── documents.py # canonical report parsing
├── config.py
└── cli.py
```

```text
raw report Markdown
        |
        v
frontmatter + body parser
        |
        v
Luna direct-target planner
        |
        v
allowlisted Supabase entity resolver
        |
        v
conditional identity escalation (Luna -> Sol)
        |
        v
Terra contextual-evidence planner
        |
        v
typed structured enrichment through existing read-only NFL tools
        |
        v
branch-scoped executor (independent targets + bounded context)
        |
        v
SQLite index (keyword text + cached vectors)
        |
        +--> keyword search (BM25)
        +--> vector search (cosine similarity)
        `--> hybrid search (reciprocal-rank fusion)
                    |
                    v
          optional batched reranker
                    |
                    v
       deterministic evidence composer
```

`ReportRetrievalPipeline` is the model-facing boundary used by
`tools/reports.py`. It always runs planned hybrid retrieval and reranking, then
returns a compact evidence contract. `strong` evidence becomes `ready`,
`partial` remains available with its coverage warning, and `weak` evidence is
returned as `no_evidence` with no report records. Documents judged irrelevant
never consume a reranker output slot and are never passed to the answering
agent. Result limits are ceilings, so fewer reports are returned when the
remaining candidates are irrelevant. The CLI intentionally remains useful for
inspecting raw retrieval and reranker diagnostics beneath that gate.

The current store is local SQLite plus cached embeddings. It is injected behind
the pipeline boundary so documents, chunks, and vector search can later move to
Supabase/pgvector without changing the `search_reports(query, limit)` tool
contract.

The generated SQLite file lives under `data/processed/` and is already ignored
by Git. Unchanged report embeddings and repeated query embeddings are reused.

Run these commands from `backend/`:

```bash
# Free/local first pass: build and test keyword retrieval.
python -m rag.cli index
python -m rag.cli search "What is the latest Kenyon Sadiq injury update?"
python -m rag.cli evaluate

# Add semantic retrieval. This reads OPENAI_API_KEY from the root .env.
python -m rag.cli index --with-embeddings
python -m rag.cli search "Who gains opportunity if Price misses time?" --mode hybrid

# Inspect both planning stages, then compare unplanned and planned retrieval.
python -m rag.cli plan "What changed in the Seahawks backfield after July 25?"
python -m rag.cli search "What changed in the Seahawks backfield after July 25?" --mode hybrid --use-planner --show-plan --show-resolution

# Retrieve 20 candidates, rerank them in one Luna call, and show why the final
# five were selected. --rerank automatically enables the planner.
python -m rag.cli search "What changed in the Seahawks backfield after July 25?" --mode hybrid --rerank --show-rerank

# Exercise and inspect nickname routing; --no-escalation shows Luna alone.
python -m rag.cli plan "How is the Sun God looking in camp?" --refresh-plan --refresh-escalation
python -m rag.cli plan "How is the Sun God looking in camp?" --no-escalation

# Isolate direct planning or refresh just the contextual stage.
python -m rag.cli plan "How is the Sun God looking in camp?" --no-context-planner
python -m rag.cli plan "How is the Sun God looking in camp?" --refresh-context

# Optional metadata filters.
python -m rag.cli search "hamstring practice" --player "Makai Lemon" --season 2026
python -m rag.cli status
```

`--with-embeddings` only embeds new or changed reports. Vector and hybrid
queries call the embeddings API once per new query; identical queries are
served from the local query cache.

`--rerank` expands retrieval to a configurable candidate pool (20 by default),
sends the question, finalized plan, resolved entities, and compact candidate
records to one structured-output model call, then returns the requested
`--limit`. Use `--rerank-candidates`, `--rerank-model`, and `--refresh-rerank`
to override those defaults. The model judges relevance, evidence relationship,
temporal role, explicit condition alignment (`supports`, `refutes`, `mixed`, or
`not_applicable`), and overall evidence sufficiency. Condition alignment is
currently exposed and logged for calibration but does not alter ranking scores.
Code—not the model—validates
document IDs and composes final results so timelines retain endpoints and order
the selected evidence chronologically, while per-entity searches retain subject
coverage. Invalid responses or API failures
fall back to the original retrieval order.

Reranker responses are cached by the complete question/plan/entity/candidate
payload, model, and prompt version. `python -m rag.cli status` reports cache use,
tokens, latency, rank changes, errors, evidence-sufficiency counts, and condition
alignment counts. Token
counts are always recorded. Dollar estimates appear only when current rates are
configured with `OPENAI_RERANK_INPUT_COST_PER_MILLION`,
`OPENAI_RERANK_CACHED_INPUT_COST_PER_MILLION`, and
`OPENAI_RERANK_OUTPUT_COST_PER_MILLION`.

Planning is deliberately staged. Luna's direct planner returns separate keyword
and semantic queries, typed entity selectors, explicit constraints, and bounded
target enrichment. Its schema contains no contextual branch field. A read-only
resolver then maps those selectors to canonical Supabase player/team records and
the existing identity router handles uncertain names. Terra receives the
original question, the identity-adjusted direct plan, and compact grounded
resolution. Its separate schema can return only indirect context branches; it
cannot recreate or mutate direct targets. Both stages cache identical inputs for
the current date. If Terra returns a malformed context plan or one that cannot
be merged with Luna's fixed direct plan, it receives the validation error and
exactly one correction attempt. API/transport failures and merely weak plans do
not trigger that retry. Planner output never becomes arbitrary SQL.

Each selector separates `hard_filters` from `soft_filters`. Hard filters must be
explicitly stated or unambiguously normalized/entailed by the question and are
the only planner filters allowed to constrain Supabase or report lookup. Soft
filters and `soft_team_mentions` preserve optional inferred context for
diagnostics and reranking but can never exclude an entity or document. Ordinary
`team_mentions` are reserved for teams actually supplied by the question.

Finite database vocabularies live in `planning/schema_values.py`. The generated planner
schema enumerates canonical team codes, player/depth-chart positions, position
groups, roster statuses, formations, ECR formats, conferences, and divisions.
Player names and other genuinely open-ended values remain strings. Current Rams
data uses `LA`; genuine relocation-era codes remain available for old seasons.

The executor links grounded player IDs to report metadata and retrieves every
selector as an independent target branch. Filters inside one selector are
combined for that target, while separate selectors are unioned before
reranking. This prevents an unrelated player and team in a comparison from
becoming an impossible global AND filter. Manual CLI filters remain explicit
global constraints and are intersected with each branch.

When useful evidence may omit a target's name, the Terra stage can add a bounded
context branch with a typed relation (`same_team`, `environment`, `dependency`,
`matchup`, or `comparison`). It runs for every planned report query and may
explicitly return that no context is needed. Its scope is explicit and local:
grounded anchor teams, anchor plus lookup-derived teams, bounded lookup-derived
entities, or a deliberately semantic-only branch. A context scope never mutates
its target or another context branch. A context anchor may be a uniquely
resolved player, a team, or an objectively filtered player group; unconstrained
and truncated groups fail before retrieval, and grouped anchors cannot fan out
into player-specific lookups. The two outputs merge into the existing
`QueryPlan` only after deterministic cross-plan validation.

`planning/lookups.py` defines the planner's allowlisted structured operations
for rosters, depth charts, schedules, player/team statistics, snap counts, ECR,
and safe formula rankings. `planning/enrichment.py` supplies grounded player IDs
and teams to the existing read-only NFL tool functions, caps returned rows and
facts, and records purpose and status. The model cannot supply internal IDs,
table names, SQL, or arbitrary tool arguments. Legal fields and finite values
are enumerated from the same processed-data catalogs used by the public tools.

Lookup purpose is executable provenance: a required relationship must resolve,
and a lookup-derived entity/team scope must actually produce that scope, before
its context branch can run. An optional candidate lookup may fail without
suppressing an independently grounded anchor-team branch. Query-enrichment
lookups may add verified terms; reranker-only lookups supply compact context
without changing search queries. Teams newly discovered by a successful
relationship-resolution lookup automatically expand that branch's local scope,
even if the proposed scope policy named only the anchor team; this prevents a
resolved opponent or counterparty from being discarded before retrieval.
Current-season depth-chart requests also retry once against the latest
`week=null` snapshot when the requested weekly row set is empty. Historical
season requests never use that fallback. Current-season roster requests with no
weekly rows fall back to `player_status` only where `last_season` matches the
requested season; the response labels that source and warns that it may be a
preseason-sized roster. Historical rosters never use the current snapshot, and
telemetry records when either fallback occurs.
The reranker receives these facts as structured context, not report evidence.
CLI and agent telemetry expose per-branch counts plus lookup operation, purpose,
status, entity count, and errors.

For every single-player reference, the planner preserves the exact phrase and
returns its best official full-name candidate, a 0–1 identity confidence, and an
enumerated resolution basis: `exact_name`, `known_alias`, `contextual_alias`, or
`inferred`. Player groups use `not_applicable`, zero confidence, and no name.
When the literal reference resembles a full name, resolution searches it
alongside Luna's canonical candidate. This preserves database-significant
spelling and punctuation distinctions without broadening short nickname lookups.

The router combines those signals with database resolution. Exact names must
agree with the uniquely matched official name; known aliases currently require
0.70 confidence, contextual aliases require 0.70, and inferred identities always
escalate. Missing/multiple database matches and malformed empty candidates also
escalate regardless of confidence. Sol receives the literal phrase, Luna's best
guess, confidence, basis, structured context, database status/matches/errors,
and trigger reasons. Multiple player references receive independent decisions
inside the same escalation call. Database candidate records include stable player
IDs and useful context when there are eight or fewer matches; larger fuzzy sets
send only their count. Duplicate-name choices are grounded by player ID, while a
new canonical name is accepted only after a fresh unique database lookup. Sol
cannot rewrite the rest of the plan, and failures keep the original Luna plan
instead of recursing.

Escalation decisions are cached in the local SQLite index. Every routed planner
execution also records whether escalation triggered, whether an API call or cache
hit occurred, token use, estimated cost, whether the plan changed, and whether
the resolved identity actually changed. Only the latter counts as identity
impact. It also persists Luna's candidate, confidence, basis, database outcome,
and routing reasons for accepted and escalated identities. `python -m rag.cli
status` summarizes signal counts and average confidence by basis alongside the
cumulative model telemetry. Cost
rates can be updated without code changes through
`OPENAI_ESCALATION_INPUT_COST_PER_MILLION`,
`OPENAI_ESCALATION_CACHED_INPUT_COST_PER_MILLION`, and
`OPENAI_ESCALATION_OUTPUT_COST_PER_MILLION`; the model can be changed with
`OPENAI_ESCALATION_MODEL`.

Telemetry is currently cumulative across routing-policy and prompt versions.
Reset or separately version the local event history before using aggregate
trigger and impact rates to calibrate the current policy.

Currently supported selector categories include player identity and career,
current roster and depth-chart role, ECR, selected season usage totals, snap
participation and opponent, plus team conference/division. Subjective phrases
such as "expected starter" remain semantic qualifiers for report search.

The package is separated by responsibility: `planning/` owns query planning,
database resolution, and identity escalation; `retrieval/` owns indexing,
embeddings, plan execution, and reranking; `evaluation/` owns regression cases
and runners. `cli.py`, `config.py`, and `documents.py` remain the shared entry,
configuration, and canonical report boundaries.

## Intentionally deferred

- Longer licensed reports will need token-aware chunks with overlap.
- The planner's target metadata remains player/team based because those are the
  entities currently stored on report documents. New metadata dimensions should
  be added only alongside ingestion/index support, not as planner-only fields.
- Structured lookups execute sequentially. Parallel execution is a latency-only
  optimization and should wait until the shared Supabase client is verified safe
  for concurrent use; it does not reduce model calls or tokens.
- The evidence gate and reranker score policy were not retuned during this
  rewrite. Branch behavior changed enough that thresholds should be calibrated
  against fresh labeled results rather than copied from the old candidate mix.
- Context expansion is one bounded planning layer, not recursive autonomous
  research. More than one dependency hop needs an explicit evaluated design.
- Required unresolved metadata does not yet produce a dedicated clarification
  status. The tool exposes unresolved constraints and withholds weak evidence,
  but broader punt/degraded behavior still needs calibration.
- The same interface can later move from local cosine search to Supabase
  Postgres/pgvector without changing the report ingestion format.
