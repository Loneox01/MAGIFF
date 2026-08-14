# Local report retrieval

This is the first, deliberately small RAG layer. It parses the report Markdown
records, treats each short report as one chunk, indexes text with SQLite FTS5,
and optionally caches OpenAI embeddings for local cosine search.

```text
rag/
├── planning/    # query plans, database resolution, identity routing
├── retrieval/   # index, embeddings, plan execution, reranking
├── evaluation/  # regression cases and runners
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
optional structured query planner
        |
        v
allowlisted Supabase entity resolver
        |
        v
conditional identity escalation (Luna -> Sol)
        |
        v
plan executor (metadata + dates + per-entity evidence)
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

# Inspect query cleanup, then compare unplanned and planned retrieval.
python -m rag.cli plan "What changed in the Seahawks backfield after July 25?"
python -m rag.cli search "What changed in the Seahawks backfield after July 25?" --mode hybrid --use-planner --show-plan --show-resolution

# Retrieve 15 candidates, rerank them in one Luna call, and show why the final
# five were selected. --rerank automatically enables the planner.
python -m rag.cli search "What changed in the Seahawks backfield after July 25?" --mode hybrid --rerank --show-rerank

# Exercise and inspect nickname routing; --no-escalation shows Luna alone.
python -m rag.cli plan "How is the Sun God looking in camp?" --refresh-plan --refresh-escalation
python -m rag.cli plan "How is the Sun God looking in camp?" --no-escalation

# Optional metadata filters.
python -m rag.cli search "hamstring practice" --player "Makai Lemon" --season 2026
python -m rag.cli status
```

`--with-embeddings` only embeds new or changed reports. Vector and hybrid
queries call the embeddings API once per new query; identical queries are
served from the local query cache.

`--rerank` expands retrieval to a configurable candidate pool (15 by default),
sends the question, finalized plan, resolved entities, and compact candidate
records to one structured-output model call, then returns the requested
`--limit`. Use `--rerank-candidates`, `--rerank-model`, and `--refresh-rerank`
to override those defaults. The model judges relevance, evidence relationship,
temporal role, and overall evidence sufficiency. Code—not the model—validates
document IDs and composes final results so timelines retain endpoints and
per-entity searches retain subject coverage. Invalid responses or API failures
fall back to the original retrieval order.

Reranker responses are cached by the complete question/plan/entity/candidate
payload, model, and prompt version. `python -m rag.cli status` reports cache use,
tokens, latency, rank changes, errors, and evidence-sufficiency counts. Token
counts are always recorded. Dollar estimates appear only when current rates are
configured with `OPENAI_RERANK_INPUT_COST_PER_MILLION`,
`OPENAI_RERANK_CACHED_INPUT_COST_PER_MILLION`, and
`OPENAI_RERANK_OUTPUT_COST_PER_MILLION`.

The planner also caches identical questions for the current date. It returns
separate keyword and semantic queries plus typed entity selectors. A read-only
resolver maps those selectors to canonical Supabase player/team records through
an allowlist of verified columns; planner output never becomes arbitrary SQL.

Finite database vocabularies live in `planning/schema_values.py`. The generated planner
schema enumerates canonical team codes, player/depth-chart positions, position
groups, roster statuses, formations, ECR formats, conferences, and divisions.
Player names and other genuinely open-ended values remain strings. Current Rams
data uses `LA`; genuine relocation-era codes remain available for old seasons.

The executor links resolved player IDs to report metadata, applies season/team/
player/date constraints, selects a baseline for change-over-time questions, and
retrieves evidence separately for comparisons. Manual CLI filters continue to
work and are combined with planned constraints.

For every single-player reference, the planner preserves the exact phrase and
returns its best official full-name candidate, a 0–1 identity confidence, and an
enumerated resolution basis: `exact_name`, `known_alias`, `contextual_alias`, or
`inferred`. Player groups use `not_applicable`, zero confidence, and no name.

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
- Required unresolved metadata does not yet halt execution with a clarification
  response. Add that abstention gate before treating retrieval as trustworthy.
- The reranker labels weak evidence but does not yet suppress it. A calibrated
  weak-result/abstention policy and answer generation remain separate next steps.
- The same interface can later move from local cosine search to Supabase
  Postgres/pgvector without changing the report ingestion format.
