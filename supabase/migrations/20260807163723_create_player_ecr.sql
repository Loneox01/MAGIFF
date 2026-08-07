-- Store time-dependent expert consensus rankings separately from player
-- identity. Current snapshots and selected final-preseason snapshots share the
-- same schema and are distinguished by snapshot_type.

create table public.player_ecr (
    player_id uuid not null references public.players(player_id) on delete cascade,
    season integer not null,
    scrape_date date not null,
    scoring_format text not null,
    league_format text not null,
    snapshot_type text not null,
    overall_rank double precision not null,
    best_rank integer,
    worst_rank integer,
    rank_sd double precision,
    rank_delta integer,
    position text,
    team text,
    source text not null,

    primary key (
        player_id,
        season,
        scrape_date,
        scoring_format,
        league_format
    ),

    constraint player_ecr_snapshot_type_check
        check (snapshot_type in ('current', 'final_preseason')),
    constraint player_ecr_scoring_format_check
        check (scoring_format in ('ppr', 'half_ppr', 'standard')),
    constraint player_ecr_positive_rank_check
        check (overall_rank > 0)
);

create index player_ecr_latest_rank_idx
    on public.player_ecr (
        season,
        scoring_format,
        league_format,
        scrape_date desc,
        overall_rank
    );

create index player_ecr_player_history_idx
    on public.player_ecr (player_id, season, scrape_date desc);

-- Backend secret-key clients can access this table; no anonymous policy is
-- created yet.
alter table public.player_ecr enable row level security;
