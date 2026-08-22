-- Quota-aware run ledger and short lease for scheduled report ingestion.
--
-- The application reserves one provider request atomically before calling the
-- FantasyPros API. This prevents overlapping GitHub Actions/manual runs from
-- exceeding the configured budget. The hard database ceiling is 50 requests
-- per UTC day; the deployed command intentionally uses a lower default of 40.

create table public.report_ingestion_runs (
    run_id uuid primary key default gen_random_uuid(),
    provider text not null,
    trigger text not null,
    status text not null,
    request_date date not null,
    requested_reports integer not null,
    provider_request_count integer not null default 0,
    daily_request_budget integer not null,

    provider_items_received integer not null default 0,
    new_reports integer not null default 0,
    changed_reports integer not null default 0,
    unchanged_reports integer not null default 0,
    failed_reports integer not null default 0,
    unresolved_player_mentions integer not null default 0,
    metadata_input_tokens integer not null default 0,
    metadata_cached_input_tokens integer not null default 0,
    metadata_output_tokens integer not null default 0,
    generated_embeddings integer not null default 0,
    reused_embeddings integer not null default 0,

    oldest_published_at timestamp with time zone,
    newest_published_at timestamp with time zone,
    feed_window_saturated boolean not null default false,
    possible_coverage_gap boolean not null default false,
    error text,
    metadata jsonb not null default '{}'::jsonb,
    started_at timestamp with time zone not null default now(),
    completed_at timestamp with time zone,

    constraint report_ingestion_runs_provider_check check (btrim(provider) <> ''),
    constraint report_ingestion_runs_trigger_check check (btrim(trigger) <> ''),
    constraint report_ingestion_runs_status_check check (
        status in ('running', 'succeeded', 'partial', 'failed', 'skipped')
    ),
    constraint report_ingestion_runs_requested_check check (requested_reports > 0),
    constraint report_ingestion_runs_request_count_check check (
        provider_request_count between 0 and 1
    ),
    constraint report_ingestion_runs_budget_check check (
        daily_request_budget between 1 and 50
    ),
    constraint report_ingestion_runs_nonnegative_counts_check check (
        provider_items_received >= 0
        and new_reports >= 0
        and changed_reports >= 0
        and unchanged_reports >= 0
        and failed_reports >= 0
        and unresolved_player_mentions >= 0
        and metadata_input_tokens >= 0
        and metadata_cached_input_tokens >= 0
        and metadata_output_tokens >= 0
        and generated_embeddings >= 0
        and reused_embeddings >= 0
    )
);

create index report_ingestion_runs_provider_day_idx
    on public.report_ingestion_runs (provider, request_date, started_at desc);
create index report_ingestion_runs_status_started_idx
    on public.report_ingestion_runs (status, started_at desc);

create table public.report_ingestion_leases (
    provider text primary key,
    run_id uuid not null references public.report_ingestion_runs(run_id)
        on delete cascade,
    acquired_at timestamp with time zone not null,
    expires_at timestamp with time zone not null,

    constraint report_ingestion_leases_provider_check check (btrim(provider) <> ''),
    constraint report_ingestion_leases_range_check check (expires_at > acquired_at)
);

alter table public.report_ingestion_runs enable row level security;
alter table public.report_ingestion_leases enable row level security;
revoke all on table public.report_ingestion_runs from anon, authenticated;
revoke all on table public.report_ingestion_leases from anon, authenticated;
grant select on table public.report_ingestion_runs to service_role;
grant select on table public.report_ingestion_leases to service_role;

create or replace function public.reserve_report_ingestion_run(
    p_provider text,
    p_trigger text,
    p_requested_reports integer,
    p_daily_request_budget integer,
    p_lease_seconds integer default 1800
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_now timestamp with time zone := now();
    v_request_date date := (v_now at time zone 'UTC')::date;
    v_run_id uuid;
    v_requests_used integer;
    v_active_run_id uuid;
begin
    if nullif(btrim(p_provider), '') is null then
        raise exception 'p_provider is required';
    end if;
    if nullif(btrim(p_trigger), '') is null then
        raise exception 'p_trigger is required';
    end if;
    if p_requested_reports < 1 then
        raise exception 'p_requested_reports must be positive';
    end if;
    if p_daily_request_budget < 1 or p_daily_request_budget > 50 then
        raise exception 'p_daily_request_budget must be between 1 and 50';
    end if;
    if p_lease_seconds < 60 or p_lease_seconds > 3600 then
        raise exception 'p_lease_seconds must be between 60 and 3600';
    end if;

    -- Serialize reservations per provider without holding a lock during network
    -- calls. The persisted lease covers the rest of the refresh.
    perform pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('report-ingestion:' || p_provider, 0)
    );

    update public.report_ingestion_runs as run
       set status = 'failed',
           error = 'The ingestion lease expired before the run was finalized.',
           completed_at = v_now
     where run.status = 'running'
       and run.run_id in (
           select lease.run_id
           from public.report_ingestion_leases as lease
           where lease.provider = p_provider and lease.expires_at <= v_now
       );

    delete from public.report_ingestion_leases
    where provider = p_provider and expires_at <= v_now;

    select run_id
      into v_active_run_id
      from public.report_ingestion_leases
     where provider = p_provider;

    -- A rolling window is stricter than assuming a provider reset timezone and
    -- remains safe around UTC-day boundaries.
    select coalesce(sum(provider_request_count), 0)::integer
      into v_requests_used
      from public.report_ingestion_runs
     where provider = p_provider
       and started_at > v_now - interval '24 hours';

    if v_active_run_id is not null then
        insert into public.report_ingestion_runs (
            provider,
            trigger,
            status,
            request_date,
            requested_reports,
            daily_request_budget,
            error,
            completed_at
        ) values (
            p_provider,
            p_trigger,
            'skipped',
            v_request_date,
            p_requested_reports,
            p_daily_request_budget,
            'Another provider refresh currently holds the ingestion lease.',
            v_now
        ) returning run_id into v_run_id;

        return jsonb_build_object(
            'acquired', false,
            'reason', 'overlap',
            'run_id', v_run_id,
            'active_run_id', v_active_run_id,
            'requests_used_last_24_hours', v_requests_used,
            'daily_request_budget', p_daily_request_budget
        );
    end if;

    if v_requests_used >= p_daily_request_budget then
        insert into public.report_ingestion_runs (
            provider,
            trigger,
            status,
            request_date,
            requested_reports,
            daily_request_budget,
            error,
            completed_at
        ) values (
            p_provider,
            p_trigger,
            'skipped',
            v_request_date,
            p_requested_reports,
            p_daily_request_budget,
            'The configured daily provider-request budget is exhausted.',
            v_now
        ) returning run_id into v_run_id;

        return jsonb_build_object(
            'acquired', false,
            'reason', 'daily_budget_exhausted',
            'run_id', v_run_id,
            'requests_used_last_24_hours', v_requests_used,
            'daily_request_budget', p_daily_request_budget
        );
    end if;

    insert into public.report_ingestion_runs (
        provider,
        trigger,
        status,
        request_date,
        requested_reports,
        provider_request_count,
        daily_request_budget
    ) values (
        p_provider,
        p_trigger,
        'running',
        v_request_date,
        p_requested_reports,
        1,
        p_daily_request_budget
    ) returning run_id into v_run_id;

    insert into public.report_ingestion_leases (
        provider,
        run_id,
        acquired_at,
        expires_at
    ) values (
        p_provider,
        v_run_id,
        v_now,
        v_now + pg_catalog.make_interval(secs => p_lease_seconds)
    );

    return jsonb_build_object(
        'acquired', true,
        'run_id', v_run_id,
        'requests_used_last_24_hours', v_requests_used + 1,
        'daily_request_budget', p_daily_request_budget,
        'lease_expires_at', v_now + pg_catalog.make_interval(secs => p_lease_seconds)
    );
end;
$$;

create or replace function public.finish_report_ingestion_run(
    p_run_id uuid,
    p_status text,
    p_metrics jsonb default '{}'::jsonb,
    p_error text default null
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_provider text;
begin
    if p_status not in ('succeeded', 'partial', 'failed') then
        raise exception 'p_status must be succeeded, partial, or failed';
    end if;
    if jsonb_typeof(coalesce(p_metrics, '{}'::jsonb)) <> 'object' then
        raise exception 'p_metrics must be a JSON object';
    end if;

    update public.report_ingestion_runs
       set status = p_status,
           provider_items_received = coalesce(
               nullif(p_metrics ->> 'provider_items_received', '')::integer,
               provider_items_received
           ),
           new_reports = coalesce(
               nullif(p_metrics ->> 'new_reports', '')::integer,
               new_reports
           ),
           changed_reports = coalesce(
               nullif(p_metrics ->> 'changed_reports', '')::integer,
               changed_reports
           ),
           unchanged_reports = coalesce(
               nullif(p_metrics ->> 'unchanged_reports', '')::integer,
               unchanged_reports
           ),
           failed_reports = coalesce(
               nullif(p_metrics ->> 'failed_reports', '')::integer,
               failed_reports
           ),
           unresolved_player_mentions = coalesce(
               nullif(p_metrics ->> 'unresolved_player_mentions', '')::integer,
               unresolved_player_mentions
           ),
           metadata_input_tokens = coalesce(
               nullif(p_metrics ->> 'metadata_input_tokens', '')::integer,
               metadata_input_tokens
           ),
           metadata_cached_input_tokens = coalesce(
               nullif(p_metrics ->> 'metadata_cached_input_tokens', '')::integer,
               metadata_cached_input_tokens
           ),
           metadata_output_tokens = coalesce(
               nullif(p_metrics ->> 'metadata_output_tokens', '')::integer,
               metadata_output_tokens
           ),
           generated_embeddings = coalesce(
               nullif(p_metrics ->> 'generated_embeddings', '')::integer,
               generated_embeddings
           ),
           reused_embeddings = coalesce(
               nullif(p_metrics ->> 'reused_embeddings', '')::integer,
               reused_embeddings
           ),
           oldest_published_at = coalesce(
               nullif(p_metrics ->> 'oldest_published_at', '')::timestamp with time zone,
               oldest_published_at
           ),
           newest_published_at = coalesce(
               nullif(p_metrics ->> 'newest_published_at', '')::timestamp with time zone,
               newest_published_at
           ),
           feed_window_saturated = coalesce(
               nullif(p_metrics ->> 'feed_window_saturated', '')::boolean,
               feed_window_saturated
           ),
           possible_coverage_gap = coalesce(
               nullif(p_metrics ->> 'possible_coverage_gap', '')::boolean,
               possible_coverage_gap
           ),
           error = p_error,
           metadata = metadata || coalesce(p_metrics, '{}'::jsonb),
           completed_at = now()
     where run_id = p_run_id
       and status = 'running'
     returning provider into v_provider;

    if v_provider is null then
        raise exception 'No running ingestion run found for %', p_run_id;
    end if;

    delete from public.report_ingestion_leases
     where provider = v_provider and run_id = p_run_id;

    return jsonb_build_object(
        'run_id', p_run_id,
        'provider', v_provider,
        'status', p_status
    );
end;
$$;

revoke all on function public.reserve_report_ingestion_run(
    text, text, integer, integer, integer
) from public, anon, authenticated;
revoke all on function public.finish_report_ingestion_run(
    uuid, text, jsonb, text
) from public, anon, authenticated;
grant execute on function public.reserve_report_ingestion_run(
    text, text, integer, integer, integer
) to service_role;
grant execute on function public.finish_report_ingestion_run(
    uuid, text, jsonb, text
) to service_role;
