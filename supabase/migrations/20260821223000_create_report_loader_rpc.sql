-- Atomically replace the current normalized representation of one report.
-- The service-role loader sends one current report, one immutable version,
-- the complete current player-link set, and the complete current chunk set.
-- Any error rolls the whole report update back.

create or replace function public.upsert_report_document(
    p_report jsonb,
    p_version jsonb,
    p_players jsonb default '[]'::jsonb,
    p_chunks jsonb default '[]'::jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
    v_report_id text := nullif(btrim(p_report ->> 'report_id'), '');
    v_version_inserted integer := 0;
begin
    if v_report_id is null then
        raise exception 'p_report.report_id is required';
    end if;
    if jsonb_typeof(coalesce(p_players, '[]'::jsonb)) <> 'array' then
        raise exception 'p_players must be a JSON array';
    end if;
    if jsonb_typeof(coalesce(p_chunks, '[]'::jsonb)) <> 'array' then
        raise exception 'p_chunks must be a JSON array';
    end if;
    if nullif(p_version ->> 'report_id', '') is distinct from v_report_id then
        raise exception 'p_version.report_id must equal p_report.report_id';
    end if;

    insert into public.reports (
        report_id,
        provider,
        external_id,
        source,
        source_url,
        title,
        author,
        language,
        published_at,
        source_updated_at,
        first_seen_at,
        last_seen_at,
        event_start_at,
        event_end_at,
        event_time_confidence,
        season,
        document_type,
        document_type_confidence,
        storyline,
        content_mode,
        source_team_id,
        teams,
        source_categories,
        body,
        source_content_hash,
        content_hash,
        metadata,
        is_active,
        retracted_at
    ) values (
        v_report_id,
        p_report ->> 'provider',
        p_report ->> 'external_id',
        p_report ->> 'source',
        p_report ->> 'source_url',
        p_report ->> 'title',
        nullif(p_report ->> 'author', ''),
        coalesce(nullif(p_report ->> 'language', ''), 'en'),
        (p_report ->> 'published_at')::timestamp with time zone,
        nullif(p_report ->> 'source_updated_at', '')::timestamp with time zone,
        (p_report ->> 'first_seen_at')::timestamp with time zone,
        (p_report ->> 'last_seen_at')::timestamp with time zone,
        nullif(p_report ->> 'event_start_at', '')::timestamp with time zone,
        nullif(p_report ->> 'event_end_at', '')::timestamp with time zone,
        nullif(p_report ->> 'event_time_confidence', '')::real,
        nullif(p_report ->> 'season', '')::integer,
        p_report ->> 'document_type',
        nullif(p_report ->> 'document_type_confidence', '')::real,
        nullif(p_report ->> 'storyline', ''),
        p_report ->> 'content_mode',
        nullif(p_report ->> 'source_team_id', ''),
        array(
            select jsonb_array_elements_text(
                coalesce(p_report -> 'teams', '[]'::jsonb)
            )
        ),
        array(
            select jsonb_array_elements_text(
                coalesce(p_report -> 'source_categories', '[]'::jsonb)
            )
        ),
        p_report ->> 'body',
        p_report ->> 'source_content_hash',
        p_report ->> 'content_hash',
        coalesce(p_report -> 'metadata', '{}'::jsonb),
        coalesce((p_report ->> 'is_active')::boolean, true),
        nullif(p_report ->> 'retracted_at', '')::timestamp with time zone
    )
    on conflict (report_id) do update set
        provider = excluded.provider,
        external_id = excluded.external_id,
        source = excluded.source,
        source_url = excluded.source_url,
        title = excluded.title,
        author = excluded.author,
        language = excluded.language,
        published_at = excluded.published_at,
        source_updated_at = excluded.source_updated_at,
        first_seen_at = least(public.reports.first_seen_at, excluded.first_seen_at),
        last_seen_at = greatest(public.reports.last_seen_at, excluded.last_seen_at),
        event_start_at = excluded.event_start_at,
        event_end_at = excluded.event_end_at,
        event_time_confidence = excluded.event_time_confidence,
        season = excluded.season,
        document_type = excluded.document_type,
        document_type_confidence = excluded.document_type_confidence,
        storyline = excluded.storyline,
        content_mode = excluded.content_mode,
        source_team_id = excluded.source_team_id,
        teams = excluded.teams,
        source_categories = excluded.source_categories,
        body = excluded.body,
        source_content_hash = excluded.source_content_hash,
        content_hash = excluded.content_hash,
        metadata = excluded.metadata,
        is_active = excluded.is_active,
        retracted_at = excluded.retracted_at;

    insert into public.report_versions (
        report_id,
        source_content_hash,
        content_hash,
        fetched_at,
        processed_at,
        metadata_model,
        metadata_prompt_version,
        normalizer_version,
        raw_payload,
        raw_storage_path,
        normalized_payload
    ) values (
        v_report_id,
        p_version ->> 'source_content_hash',
        p_version ->> 'content_hash',
        (p_version ->> 'fetched_at')::timestamp with time zone,
        (p_version ->> 'processed_at')::timestamp with time zone,
        nullif(p_version ->> 'metadata_model', ''),
        nullif(p_version ->> 'metadata_prompt_version', ''),
        p_version ->> 'normalizer_version',
        p_version -> 'raw_payload',
        nullif(p_version ->> 'raw_storage_path', ''),
        p_version -> 'normalized_payload'
    )
    on conflict (report_id, source_content_hash, content_hash) do nothing;
    get diagnostics v_version_inserted = row_count;

    -- Player metadata can change without a content change, so replace the
    -- small current relationship set on every load.
    delete from public.report_players
    where report_id = v_report_id;

    insert into public.report_players (
        report_id,
        player_id,
        reference_text,
        identity_confidence,
        resolution_basis,
        mention_role,
        resolution_source
    )
    select
        v_report_id,
        (player ->> 'player_id')::uuid,
        player ->> 'reference_text',
        (player ->> 'identity_confidence')::real,
        player ->> 'resolution_basis',
        player ->> 'mention_role',
        player ->> 'resolution_source'
    from jsonb_array_elements(coalesce(p_players, '[]'::jsonb)) as player;

    -- The report update trigger already removes chunks after a content change.
    -- For unchanged content, a NULL incoming embedding preserves the existing
    -- vector, avoiding repeat embedding calls on idempotent loads.
    insert into public.report_chunks (
        chunk_id,
        report_id,
        chunk_index,
        heading,
        content,
        embedding_text,
        content_hash,
        token_count,
        chunk_metadata,
        embedding,
        embedding_model,
        embedded_at
    )
    select
        chunk ->> 'chunk_id',
        v_report_id,
        (chunk ->> 'chunk_index')::integer,
        nullif(chunk ->> 'heading', ''),
        chunk ->> 'content',
        chunk ->> 'embedding_text',
        chunk ->> 'content_hash',
        nullif(chunk ->> 'token_count', '')::integer,
        coalesce(chunk -> 'chunk_metadata', '{}'::jsonb),
        case
            when chunk -> 'embedding' is null
              or chunk -> 'embedding' = 'null'::jsonb then null
            else (chunk -> 'embedding')::text::extensions.vector
        end,
        nullif(chunk ->> 'embedding_model', ''),
        nullif(chunk ->> 'embedded_at', '')::timestamp with time zone
    from jsonb_array_elements(coalesce(p_chunks, '[]'::jsonb)) as chunk
    on conflict (chunk_id) do update set
        report_id = excluded.report_id,
        chunk_index = excluded.chunk_index,
        heading = excluded.heading,
        content = excluded.content,
        embedding_text = excluded.embedding_text,
        content_hash = excluded.content_hash,
        token_count = excluded.token_count,
        chunk_metadata = excluded.chunk_metadata,
        embedding = coalesce(excluded.embedding, public.report_chunks.embedding),
        embedding_model = case
            when excluded.embedding is null
                then public.report_chunks.embedding_model
            else excluded.embedding_model
        end,
        embedded_at = case
            when excluded.embedding is null
                then public.report_chunks.embedded_at
            else excluded.embedded_at
        end;

    delete from public.report_chunks as existing
    where existing.report_id = v_report_id
      and not exists (
          select 1
          from jsonb_array_elements(coalesce(p_chunks, '[]'::jsonb)) as current_chunk
          where current_chunk ->> 'chunk_id' = existing.chunk_id
      );

    return jsonb_build_object(
        'report_id', v_report_id,
        'version_inserted', v_version_inserted = 1,
        'player_count', jsonb_array_length(coalesce(p_players, '[]'::jsonb)),
        'chunk_count', jsonb_array_length(coalesce(p_chunks, '[]'::jsonb))
    );
end;
$$;

revoke execute on function public.upsert_report_document(
    jsonb,
    jsonb,
    jsonb,
    jsonb
) from public, anon, authenticated;

grant execute on function public.upsert_report_document(
    jsonb,
    jsonb,
    jsonb,
    jsonb
) to service_role;
