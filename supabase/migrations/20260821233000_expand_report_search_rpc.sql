-- Return complete report metadata from retrieval RPCs so the application can
-- switch between local SQLite and Supabase without changing the planner,
-- executor, reciprocal-rank fusion, reranker, or evidence gate.

create or replace function public.match_report_chunks_v2(
    query_embedding extensions.vector(1536),
    match_threshold double precision default 0,
    match_count integer default 20,
    filter_embedding_model text default null,
    filter_season integer default null,
    filter_teams text[] default null,
    filter_player_ids uuid[] default null,
    filter_player_names text[] default null,
    filter_source text default null,
    filter_document_type text default null,
    filter_storyline text default null,
    published_after date default null,
    published_before date default null,
    published_from date default null,
    published_to date default null
)
returns table (
    chunk_id text,
    report_id text,
    title text,
    source text,
    source_url text,
    author text,
    published_at timestamp with time zone,
    fetched_at timestamp with time zone,
    player_ids uuid[],
    player_names text[],
    teams text[],
    season integer,
    document_type text,
    storyline text,
    content_mode text,
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
        report.author,
        report.published_at,
        report.last_seen_at,
        array(
            select report_player.player_id
            from public.report_players as report_player
            where report_player.report_id = report.report_id
            order by
                case report_player.mention_role
                    when 'primary_subject' then 0
                    when 'materially_affected' then 1
                    else 2
                end,
                report_player.player_id
        ),
        array(
            select player.display_name
            from public.report_players as report_player
            join public.players as player
              on player.player_id = report_player.player_id
            where report_player.report_id = report.report_id
            order by
                case report_player.mention_role
                    when 'primary_subject' then 0
                    when 'materially_affected' then 1
                    else 2
                end,
                report_player.player_id
        ),
        report.teams,
        report.season,
        report.document_type,
        report.storyline,
        report.content_mode,
        chunk.content,
        1 - (chunk.embedding operator(extensions.<=>) query_embedding)
    from public.report_chunks as chunk
    join public.reports as report on report.report_id = chunk.report_id
    where report.is_active
      and chunk.embedding is not null
      and (
          filter_embedding_model is null
          or chunk.embedding_model = filter_embedding_model
      )
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
      and (
          filter_player_names is null
          or cardinality(filter_player_names) = 0
          or exists (
              select 1
              from public.report_players as report_player
              join public.players as player
                on player.player_id = report_player.player_id
              cross join unnest(filter_player_names) as wanted(name)
              where report_player.report_id = report.report_id
                and player.display_name ilike '%' || wanted.name || '%'
          )
      )
      and (
          filter_source is null
          or report.source ilike '%' || filter_source || '%'
      )
      and (
          filter_document_type is null
          or report.document_type = filter_document_type
      )
      and (filter_storyline is null or report.storyline = filter_storyline)
      and (published_after is null or report.published_at::date > published_after)
      and (published_before is null or report.published_at::date < published_before)
      and (published_from is null or report.published_at::date >= published_from)
      and (published_to is null or report.published_at::date <= published_to)
    order by chunk.embedding operator(extensions.<=>) query_embedding
    limit least(greatest(match_count, 1), 100);
$$;

create or replace function public.search_report_chunks_v2(
    query_text text,
    match_count integer default 20,
    filter_season integer default null,
    filter_teams text[] default null,
    filter_player_ids uuid[] default null,
    filter_player_names text[] default null,
    filter_source text default null,
    filter_document_type text default null,
    filter_storyline text default null,
    published_after date default null,
    published_before date default null,
    published_from date default null,
    published_to date default null
)
returns table (
    chunk_id text,
    report_id text,
    title text,
    source text,
    source_url text,
    author text,
    published_at timestamp with time zone,
    fetched_at timestamp with time zone,
    player_ids uuid[],
    player_names text[],
    teams text[],
    season integer,
    document_type text,
    storyline text,
    content_mode text,
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
        report.author,
        report.published_at,
        report.last_seen_at,
        array(
            select report_player.player_id
            from public.report_players as report_player
            where report_player.report_id = report.report_id
            order by
                case report_player.mention_role
                    when 'primary_subject' then 0
                    when 'materially_affected' then 1
                    else 2
                end,
                report_player.player_id
        ),
        array(
            select player.display_name
            from public.report_players as report_player
            join public.players as player
              on player.player_id = report_player.player_id
            where report_player.report_id = report.report_id
            order by
                case report_player.mention_role
                    when 'primary_subject' then 0
                    when 'materially_affected' then 1
                    else 2
                end,
                report_player.player_id
        ),
        report.teams,
        report.season,
        report.document_type,
        report.storyline,
        report.content_mode,
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
      and (
          filter_player_names is null
          or cardinality(filter_player_names) = 0
          or exists (
              select 1
              from public.report_players as report_player
              join public.players as player
                on player.player_id = report_player.player_id
              cross join unnest(filter_player_names) as wanted(name)
              where report_player.report_id = report.report_id
                and player.display_name ilike '%' || wanted.name || '%'
          )
      )
      and (
          filter_source is null
          or report.source ilike '%' || filter_source || '%'
      )
      and (
          filter_document_type is null
          or report.document_type = filter_document_type
      )
      and (filter_storyline is null or report.storyline = filter_storyline)
      and (published_after is null or report.published_at::date > published_after)
      and (published_before is null or report.published_at::date < published_before)
      and (published_from is null or report.published_at::date >= published_from)
      and (published_to is null or report.published_at::date <= published_to)
    order by keyword_rank desc, report.published_at desc
    limit least(greatest(match_count, 1), 100);
$$;

create or replace function public.report_store_status()
returns jsonb
language sql
stable
security invoker
set search_path = ''
as $$
    select jsonb_build_object(
        'store', 'supabase',
        'document_count', (
            select count(*) from public.reports where is_active
        ),
        'chunk_count', (select count(*) from public.report_chunks),
        'embedded_count', (
            select count(*) from public.report_chunks where embedding is not null
        ),
        'embedding_models', coalesce(
            (
                select jsonb_agg(models.embedding_model order by models.embedding_model)
                from (
                    select distinct embedding_model
                    from public.report_chunks
                    where embedding_model is not null
                ) as models
            ),
            '[]'::jsonb
        )
    );
$$;

revoke execute on function public.match_report_chunks_v2(
    extensions.vector,
    double precision,
    integer,
    text,
    integer,
    text[],
    uuid[],
    text[],
    text,
    text,
    text,
    date,
    date,
    date,
    date
) from public, anon, authenticated;

revoke execute on function public.search_report_chunks_v2(
    text,
    integer,
    integer,
    text[],
    uuid[],
    text[],
    text,
    text,
    text,
    date,
    date,
    date,
    date
) from public, anon, authenticated;

revoke execute on function public.report_store_status()
from public, anon, authenticated;

grant execute on function public.match_report_chunks_v2(
    extensions.vector,
    double precision,
    integer,
    text,
    integer,
    text[],
    uuid[],
    text[],
    text,
    text,
    text,
    date,
    date,
    date,
    date
) to service_role;

grant execute on function public.search_report_chunks_v2(
    text,
    integer,
    integer,
    text[],
    uuid[],
    text[],
    text,
    text,
    text,
    date,
    date,
    date,
    date
) to service_role;

grant execute on function public.report_store_status() to service_role;
