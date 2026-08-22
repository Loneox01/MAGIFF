-- Store normalized unstructured reports separately from structured NFL facts.
--
-- `reports` is the current retrieval-facing representation. Immutable source
-- and normalization revisions are retained in `report_versions`. Player
-- identity is relational rather than embedded only in JSON, while teams remain
-- a small searchable array. `report_chunks` supports both PostgreSQL full-text
-- search and 1536-dimensional text-embedding-3-small vectors.

create extension if not exists vector with schema extensions;

create schema if not exists private;
revoke all on schema private from public, anon, authenticated;

create table public.reports (
    report_id text primary key,
    provider text not null,
    external_id text not null,
    source text not null,
    source_url text not null,
    title text not null,
    author text,
    language text not null default 'en',

    published_at timestamp with time zone not null,
    source_updated_at timestamp with time zone,
    first_seen_at timestamp with time zone not null,
    last_seen_at timestamp with time zone not null,

    -- Event time is distinct from publication time. These remain NULL until a
    -- source or later enrichment can ground them confidently.
    event_start_at timestamp with time zone,
    event_end_at timestamp with time zone,
    event_time_confidence real,

    season integer,
    document_type text not null,
    document_type_confidence real,
    storyline text,
    content_mode text not null,

    source_team_id text,
    teams text[] not null default '{}',
    source_categories text[] not null default '{}',

    body text not null,
    source_content_hash text not null,
    content_hash text not null,

    -- Provider-specific and enrichment metadata that is useful for auditing
    -- but does not warrant a new relational column belongs here.
    metadata jsonb not null default '{}'::jsonb,
    is_active boolean not null default true,
    retracted_at timestamp with time zone,

    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now(),

    constraint reports_provider_external_key unique (provider, external_id),
    constraint reports_nonempty_identity_check check (
        btrim(report_id) <> ''
        and btrim(provider) <> ''
        and btrim(external_id) <> ''
    ),
    constraint reports_nonempty_content_check check (
        btrim(title) <> ''
        and btrim(source_url) <> ''
        and btrim(body) <> ''
        and btrim(source_content_hash) <> ''
        and btrim(content_hash) <> ''
    ),
    constraint reports_document_type_confidence_check check (
        document_type_confidence is null
        or document_type_confidence between 0 and 1
    ),
    constraint reports_event_time_confidence_check check (
        event_time_confidence is null
        or event_time_confidence between 0 and 1
    ),
    constraint reports_event_range_check check (
        event_start_at is null
        or event_end_at is null
        or event_end_at >= event_start_at
    ),
    constraint reports_seen_range_check check (last_seen_at >= first_seen_at),
    constraint reports_retraction_check check (
        (is_active and retracted_at is null)
        or not is_active
    )
);

-- Immutable audit history. The current normalized fields stay on `reports` for
-- simple retrieval; each distinct raw/normalized combination is preserved here.
-- Large future source payloads may live in Storage and use raw_storage_path.
create table public.report_versions (
    report_version_id bigint generated always as identity primary key,
    report_id text not null references public.reports(report_id) on delete cascade,
    source_content_hash text not null,
    content_hash text not null,
    fetched_at timestamp with time zone not null,
    processed_at timestamp with time zone not null,
    metadata_model text,
    metadata_prompt_version text,
    normalizer_version text not null,
    raw_payload jsonb,
    raw_storage_path text,
    normalized_payload jsonb not null,
    created_at timestamp with time zone not null default now(),

    constraint report_versions_content_key unique (
        report_id,
        source_content_hash,
        content_hash
    ),
    constraint report_versions_source_check check (
        raw_payload is not null or raw_storage_path is not null
    ),
    constraint report_versions_nonempty_hashes_check check (
        btrim(source_content_hash) <> '' and btrim(content_hash) <> ''
    )
);

-- Current, database-grounded player relationships used by metadata filtering.
-- Historical relationship state remains recoverable from normalized_payload in
-- report_versions when a later version changes the entity set.
create table public.report_players (
    report_id text not null references public.reports(report_id) on delete cascade,
    player_id uuid not null references public.players(player_id) on delete cascade,
    reference_text text not null,
    identity_confidence real not null,
    resolution_basis text not null,
    mention_role text not null,
    resolution_source text not null,
    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now(),

    primary key (report_id, player_id),
    constraint report_players_confidence_check check (
        identity_confidence between 0 and 1
    ),
    constraint report_players_basis_check check (
        resolution_basis in (
            'provider_id',
            'exact_name',
            'known_alias',
            'contextual_alias',
            'inferred'
        )
    ),
    constraint report_players_role_check check (
        mention_role in ('primary_subject', 'materially_affected', 'contextual')
    )
);

-- Short provider news will normally have one chunk. Keeping chunk identity and
-- ordering separate lets long-form articles, reports, and PDFs use the same
-- retrieval layer later without changing the report schema.
create table public.report_chunks (
    chunk_id text primary key,
    report_id text not null references public.reports(report_id) on delete cascade,
    chunk_index integer not null,
    heading text,
    content text not null,

    -- Includes the title and selected metadata as well as chunk content. Both
    -- FTS and the embedding model should operate on this same attributable text.
    embedding_text text not null,
    content_hash text not null,
    token_count integer,
    chunk_metadata jsonb not null default '{}'::jsonb,

    embedding extensions.vector(1536),
    embedding_model text,
    embedded_at timestamp with time zone,
    search_vector tsvector generated always as (
        to_tsvector('pg_catalog.english'::regconfig, coalesce(embedding_text, ''))
    ) stored,

    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now(),

    constraint report_chunks_report_index_key unique (report_id, chunk_index),
    constraint report_chunks_nonnegative_index_check check (chunk_index >= 0),
    constraint report_chunks_nonnegative_tokens_check check (
        token_count is null or token_count >= 0
    ),
    constraint report_chunks_nonempty_content_check check (
        btrim(chunk_id) <> ''
        and btrim(content) <> ''
        and btrim(embedding_text) <> ''
        and btrim(content_hash) <> ''
    ),
    constraint report_chunks_embedding_model_check check (
        embedding is null or embedding_model is not null
    )
);

-- Current-document filters and recency paths.
create index reports_published_idx
    on public.reports (published_at desc)
    where is_active;
create index reports_provider_published_idx
    on public.reports (provider, published_at desc)
    where is_active;
create index reports_type_published_idx
    on public.reports (document_type, published_at desc)
    where is_active;
create index reports_season_published_idx
    on public.reports (season, published_at desc)
    where is_active;
create index reports_teams_gin_idx on public.reports using gin (teams);
create index reports_categories_gin_idx
    on public.reports using gin (source_categories);
create index reports_metadata_gin_idx on public.reports using gin (metadata);

create index report_versions_report_observed_idx
    on public.report_versions (report_id, fetched_at desc);
create index report_players_player_idx
    on public.report_players (player_id, report_id);
create index report_chunks_report_idx
    on public.report_chunks (report_id, chunk_index);
create index report_chunks_search_idx
    on public.report_chunks using gin (search_vector);
create index report_chunks_embedding_hnsw_idx
    on public.report_chunks using hnsw (
        embedding extensions.vector_cosine_ops
    )
    where embedding is not null;

-- Keep update timestamps server-controlled.
create or replace function private.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create trigger reports_set_updated_at
before update on public.reports
for each row execute function private.set_updated_at();

create trigger report_players_set_updated_at
before update on public.report_players
for each row execute function private.set_updated_at();

create trigger report_chunks_set_updated_at
before update on public.report_chunks
for each row execute function private.set_updated_at();

-- A changed normalized report must never retain stale player links or stale
-- embeddings. The future loader should upsert the report and repopulate its
-- relationships/chunks in one transaction or RPC.
create or replace function private.invalidate_changed_report()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    if old.content_hash is distinct from new.content_hash then
        delete from public.report_players where report_id = new.report_id;
        delete from public.report_chunks where report_id = new.report_id;
    end if;
    return new;
end;
$$;

create trigger reports_invalidate_changed_derivatives
after update of content_hash on public.reports
for each row execute function private.invalidate_changed_report();

-- Semantic candidate retrieval for the backend. Metadata predicates stay in
-- SQL so they can be combined with app-side reciprocal-rank fusion and
-- reranking. Limits are bounded defensively for PostgREST callers.
create or replace function public.match_report_chunks(
    query_embedding extensions.vector(1536),
    match_threshold double precision default 0,
    match_count integer default 20,
    filter_season integer default null,
    filter_teams text[] default null,
    filter_player_ids uuid[] default null
)
returns table (
    chunk_id text,
    report_id text,
    title text,
    source text,
    source_url text,
    published_at timestamp with time zone,
    document_type text,
    content text,
    similarity double precision
)
language sql
stable
security invoker
set search_path = ''
as $$
    select
        chunk.chunk_id,
        report.report_id,
        report.title,
        report.source,
        report.source_url,
        report.published_at,
        report.document_type,
        chunk.content,
        1 - (chunk.embedding operator(extensions.<=>) query_embedding) as similarity
    from public.report_chunks as chunk
    join public.reports as report on report.report_id = chunk.report_id
    where report.is_active
      and chunk.embedding is not null
      and 1 - (chunk.embedding operator(extensions.<=>) query_embedding)
          >= match_threshold
      and (filter_season is null or report.season = filter_season)
      and (
          filter_teams is null
          or cardinality(filter_teams) = 0
          or report.teams && filter_teams
      )
      and (
          filter_player_ids is null
          or cardinality(filter_player_ids) = 0
          or exists (
              select 1
              from public.report_players as report_player
              where report_player.report_id = report.report_id
                and report_player.player_id = any(filter_player_ids)
          )
      )
    order by chunk.embedding operator(extensions.<=>) query_embedding
    limit least(greatest(match_count, 1), 100);
$$;

-- Keyword candidates use PostgreSQL web-search syntax and the same filters as
-- vector retrieval. Hybrid fusion remains in application code so its ranking
-- policy can evolve without a database migration.
create or replace function public.search_report_chunks(
    query_text text,
    match_count integer default 20,
    filter_season integer default null,
    filter_teams text[] default null,
    filter_player_ids uuid[] default null
)
returns table (
    chunk_id text,
    report_id text,
    title text,
    source text,
    source_url text,
    published_at timestamp with time zone,
    document_type text,
    content text,
    keyword_rank real
)
language sql
stable
security invoker
set search_path = ''
as $$
    select
        chunk.chunk_id,
        report.report_id,
        report.title,
        report.source,
        report.source_url,
        report.published_at,
        report.document_type,
        chunk.content,
        ts_rank_cd(
            chunk.search_vector,
            websearch_to_tsquery('pg_catalog.english'::regconfig, query_text)
        ) as keyword_rank
    from public.report_chunks as chunk
    join public.reports as report on report.report_id = chunk.report_id
    where report.is_active
      and btrim(query_text) <> ''
      and chunk.search_vector @@
          websearch_to_tsquery('pg_catalog.english'::regconfig, query_text)
      and (filter_season is null or report.season = filter_season)
      and (
          filter_teams is null
          or cardinality(filter_teams) = 0
          or report.teams && filter_teams
      )
      and (
          filter_player_ids is null
          or cardinality(filter_player_ids) = 0
          or exists (
              select 1
              from public.report_players as report_player
              where report_player.report_id = report.report_id
                and report_player.player_id = any(filter_player_ids)
          )
      )
    order by keyword_rank desc, report.published_at desc
    limit least(greatest(match_count, 1), 100);
$$;

-- Service-role backend only. No anonymous or authenticated-user policies are
-- created; RLS therefore blocks direct frontend access.
alter table public.reports enable row level security;
alter table public.report_versions enable row level security;
alter table public.report_players enable row level security;
alter table public.report_chunks enable row level security;

revoke all on table
    public.reports,
    public.report_versions,
    public.report_players,
    public.report_chunks
from anon, authenticated;

grant select, insert, update, delete on table
    public.reports,
    public.report_versions,
    public.report_players,
    public.report_chunks
to service_role;

grant usage, select on sequence public.report_versions_report_version_id_seq
to service_role;

revoke execute on function public.match_report_chunks(
    extensions.vector,
    double precision,
    integer,
    integer,
    text[],
    uuid[]
) from public, anon, authenticated;

revoke execute on function public.search_report_chunks(
    text,
    integer,
    integer,
    text[],
    uuid[]
) from public, anon, authenticated;

grant execute on function public.match_report_chunks(
    extensions.vector,
    double precision,
    integer,
    integer,
    text[],
    uuid[]
) to service_role;

grant execute on function public.search_report_chunks(
    text,
    integer,
    integer,
    text[],
    uuid[]
) to service_role;
