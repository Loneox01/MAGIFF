-- Persistent, idempotent audit ledger for automatic read-only lineup reviews.

create table public.lineup_review_runs (
    review_id uuid primary key default gen_random_uuid(),
    review_key text not null unique,
    league_id text not null,
    roster_id integer not null,
    season integer not null,
    week integer not null,
    slate_kickoff timestamp with time zone,
    trigger text not null,
    status text not null default 'running',
    outcome text,
    lead_minutes integer not null default 75,

    slate_players jsonb not null default '[]'::jsonb,
    health_snapshot jsonb not null default '{}'::jsonb,
    context_snapshot jsonb not null default '{}'::jsonb,
    analysis jsonb,
    changes jsonb not null default '[]'::jsonb,
    provisional_changes jsonb not null default '[]'::jsonb,
    warnings jsonb not null default '[]'::jsonb,

    model text,
    latency_seconds double precision,
    input_tokens integer not null default 0,
    cached_input_tokens integer not null default 0,
    output_tokens integer not null default 0,
    estimated_cost_usd double precision,
    error text,

    notification_status text not null default 'skipped',
    notification_message text,
    notification_error text,
    discord_message_id text,
    started_at timestamp with time zone not null default now(),
    completed_at timestamp with time zone,
    notified_at timestamp with time zone,

    constraint lineup_review_runs_review_key_check check (btrim(review_key) <> ''),
    constraint lineup_review_runs_league_check check (btrim(league_id) <> ''),
    constraint lineup_review_runs_week_check check (week between 1 and 22),
    constraint lineup_review_runs_lead_check check (lead_minutes between 1 and 180),
    constraint lineup_review_runs_trigger_check check (
        trigger in ('scheduled', 'emergency', 'e2e')
    ),
    constraint lineup_review_runs_status_check check (
        status in ('running', 'succeeded', 'failed')
    ),
    constraint lineup_review_runs_outcome_check check (
        outcome is null or outcome in (
            'no_change',
            'change_recommended',
            'review_failed',
            'emergency_update'
        )
    ),
    constraint lineup_review_runs_notification_check check (
        notification_status in ('skipped', 'pending', 'sent', 'failed')
    ),
    constraint lineup_review_runs_usage_check check (
        input_tokens >= 0
        and cached_input_tokens >= 0
        and output_tokens >= 0
    )
);

create index lineup_review_runs_slate_idx
    on public.lineup_review_runs (
        league_id,
        roster_id,
        season,
        week,
        slate_kickoff desc,
        started_at desc
    );

create index lineup_review_runs_notification_idx
    on public.lineup_review_runs (notification_status, started_at)
    where notification_status in ('pending', 'failed');

alter table public.lineup_review_runs enable row level security;
revoke all on table public.lineup_review_runs from anon, authenticated;
grant select, insert, update on table public.lineup_review_runs to service_role;

