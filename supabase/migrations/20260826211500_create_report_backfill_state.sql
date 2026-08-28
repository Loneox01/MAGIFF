-- Persist one immutable player-feed queue per historical report backfill.
-- This prevents later ECR, roster, or identity refreshes from reordering or
-- replacing the candidates of an in-progress job.

create table public.report_backfills (
    backfill_id text primary key,
    provider text not null,
    season integer not null,
    scoring_format text not null,
    league_format text not null,
    ecr_snapshot_date date not null,
    candidate_limit integer not null,
    candidates jsonb not null,
    created_at timestamp with time zone not null default now(),

    constraint report_backfills_id_check check (btrim(backfill_id) <> ''),
    constraint report_backfills_provider_check check (btrim(provider) <> ''),
    constraint report_backfills_candidate_limit_check check (candidate_limit > 0),
    constraint report_backfills_candidates_check check (
        jsonb_typeof(candidates) = 'array'
    )
);

alter table public.report_backfills enable row level security;
revoke all on table public.report_backfills from anon, authenticated;
grant select, insert on table public.report_backfills to service_role;
