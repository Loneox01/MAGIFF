-- Expand the structured NFL store for all processed reference, current, and
-- season-level datasets. Season-dependent tables are shared across years.

create table public.teams (
    team_abbr text primary key,
    team_id text not null unique,
    team_name text not null,
    team_nick text not null,
    team_conf text,
    team_division text,
    team_color text,
    team_color2 text,
    team_logo_espn text,
    team_wordmark text,
    team_logo_squared text
);

create table public.player_external_ids (
    player_id uuid not null references public.players(player_id) on delete cascade,
    provider text not null,
    external_id text not null,

    primary key (provider, external_id)
);

create index player_external_ids_player_idx
    on public.player_external_ids (player_id);

create table public.player_season_stats (
    player_id uuid not null references public.players(player_id) on delete cascade,
    gsis_id text not null,
    season integer not null,
    season_type text not null,
    games integer not null,
    teams text[] not null,
    last_team text,
    position text,
    position_group text,
    completions integer not null,
    attempts integer not null,
    passing_yards integer not null,
    passing_tds integer not null,
    passing_interceptions integer not null,
    sacks_suffered integer not null,
    sack_yards_lost integer not null,
    passing_air_yards integer not null,
    passing_yards_after_catch integer not null,
    passing_first_downs integer not null,
    passing_2pt_conversions integer not null,
    carries integer not null,
    rushing_yards integer not null,
    rushing_tds integer not null,
    rushing_fumbles integer not null,
    rushing_fumbles_lost integer not null,
    rushing_first_downs integer not null,
    rushing_2pt_conversions integer not null,
    receptions integer not null,
    targets integer not null,
    receiving_yards integer not null,
    receiving_tds integer not null,
    receiving_fumbles integer not null,
    receiving_fumbles_lost integer not null,
    receiving_air_yards integer not null,
    receiving_yards_after_catch integer not null,
    receiving_first_downs integer not null,
    receiving_2pt_conversions integer not null,
    fantasy_points double precision not null,
    fantasy_points_ppr double precision not null,
    passing_epa double precision,
    rushing_epa double precision,
    receiving_epa double precision,
    completion_percentage double precision,
    passing_yards_per_attempt double precision,
    passing_epa_per_attempt double precision,
    passing_cpoe double precision,
    pacr double precision,
    rushing_yards_per_carry double precision,
    rushing_epa_per_carry double precision,
    catch_percentage double precision,
    receiving_yards_per_reception double precision,
    receiving_yards_per_target double precision,
    receiving_epa_per_target double precision,
    racr double precision,
    fantasy_points_per_game double precision,
    fantasy_points_ppr_per_game double precision,

    primary key (player_id, season, season_type)
);

create table public.player_weekly_rosters (
    player_id uuid not null references public.players(player_id) on delete cascade,
    gsis_id text not null,
    season integer not null,
    week integer not null,
    game_type text not null,
    team text not null,
    position text,
    depth_chart_position text,
    jersey_number integer,
    status text,
    status_description_abbr text,
    years_exp integer,

    primary key (player_id, season, week, game_type, team)
);

create table public.player_snap_counts (
    player_id uuid not null references public.players(player_id) on delete cascade,
    game_id text not null references public.games(game_id) on delete cascade,
    season integer not null,
    game_type text not null,
    week integer not null,
    pfr_player_id text,
    position text,
    team text not null,
    opponent text not null,
    offense_snaps double precision,
    offense_pct double precision,
    defense_snaps double precision,
    defense_pct double precision,
    st_snaps double precision,
    st_pct double precision,

    primary key (player_id, game_id)
);

create table public.team_weekly_stats (
    season integer not null,
    week integer not null,
    season_type text not null,
    game_id text not null references public.games(game_id) on delete cascade,
    team text not null,
    opponent_team text not null,
    completions integer not null,
    attempts integer not null,
    passing_yards integer not null,
    passing_tds integer not null,
    passing_interceptions integer not null,
    sacks_suffered integer not null,
    sack_yards_lost integer not null,
    passing_air_yards integer not null,
    passing_yards_after_catch integer not null,
    passing_first_downs integer not null,
    passing_epa double precision,
    passing_cpoe double precision,
    passing_2pt_conversions integer not null,
    carries integer not null,
    rushing_yards integer not null,
    rushing_tds integer not null,
    rushing_fumbles integer not null,
    rushing_fumbles_lost integer not null,
    rushing_first_downs integer not null,
    rushing_epa double precision,
    rushing_2pt_conversions integer not null,
    receptions integer not null,
    targets integer not null,
    receiving_yards integer not null,
    receiving_tds integer not null,
    receiving_fumbles integer not null,
    receiving_fumbles_lost integer not null,
    receiving_air_yards integer not null,
    receiving_yards_after_catch integer not null,
    receiving_first_downs integer not null,
    receiving_epa double precision,
    receiving_2pt_conversions integer not null,

    primary key (team, game_id)
);

-- Completed-season depth charts: generally one pregame snapshot per team/week.
-- NULLS NOT DISTINCT lets the legacy 2024 rows, which lack timestamps and
-- numeric position slots, still participate in duplicate prevention.
create table public.depth_chart_entries (
    depth_chart_entry_id bigint generated always as identity primary key,
    player_id uuid not null references public.players(player_id) on delete cascade,
    season integer not null,
    week integer,
    season_type text,
    snapshot_at timestamp with time zone,
    team text not null,
    player_name text not null,
    gsis_id text,
    espn_id text,
    formation text,
    position_group text,
    position_name text,
    position text,
    position_slot integer,
    depth_rank integer,
    jersey_number text,

    constraint depth_chart_entries_logical_key
        unique nulls not distinct (
            season,
            season_type,
            week,
            team,
            player_id,
            formation,
            position,
            position_slot,
            depth_rank
        )
);

-- Overwriteable active-season view of the newest snapshot for each team.
create table public.current_depth_chart_entries (
    current_depth_chart_entry_id bigint generated always as identity primary key,
    player_id uuid not null references public.players(player_id) on delete cascade,
    season integer not null,
    week integer,
    season_type text,
    snapshot_at timestamp with time zone,
    team text not null,
    player_name text not null,
    gsis_id text,
    espn_id text,
    formation text,
    position_group text,
    position_name text,
    position text,
    position_slot integer,
    depth_rank integer,
    jersey_number text,

    constraint current_depth_chart_entries_logical_key
        unique nulls not distinct (
            season,
            team,
            player_id,
            formation,
            position,
            position_slot,
            depth_rank
        )
);

-- Rankings and common agent-tool lookup paths.
create index player_season_stats_passing_idx
    on public.player_season_stats (season, season_type, passing_yards desc);
create index player_season_stats_rushing_idx
    on public.player_season_stats (season, season_type, rushing_yards desc);
create index player_season_stats_receiving_idx
    on public.player_season_stats (season, season_type, receiving_yards desc);
create index player_season_stats_fantasy_idx
    on public.player_season_stats (season, season_type, fantasy_points_ppr desc);
create index player_weekly_rosters_team_week_idx
    on public.player_weekly_rosters (team, season, week);
create index player_snap_counts_team_week_idx
    on public.player_snap_counts (team, season, week);
create index team_weekly_stats_season_week_idx
    on public.team_weekly_stats (season, week, team);
create index depth_chart_entries_team_week_idx
    on public.depth_chart_entries (season, week, team);
create index depth_chart_entries_player_idx
    on public.depth_chart_entries (player_id, season, week);
create index current_depth_chart_entries_team_idx
    on public.current_depth_chart_entries (season, team);

-- Upcoming schedules contain unknown results and may have other fields that
-- are not finalized yet. Keep identifiers and matchup fields required while
-- allowing mutable schedule details to remain NULL.
alter table public.games alter column gametime drop not null;
alter table public.games alter column away_score drop not null;
alter table public.games alter column home_score drop not null;
alter table public.games alter column location drop not null;
alter table public.games alter column overtime drop not null;
alter table public.games alter column away_rest drop not null;
alter table public.games alter column home_rest drop not null;
alter table public.games alter column spread_line drop not null;
alter table public.games alter column total_line drop not null;
alter table public.games alter column roof drop not null;
alter table public.games alter column surface drop not null;
alter table public.games alter column away_qb_id drop not null;
alter table public.games alter column home_qb_id drop not null;
alter table public.games alter column away_qb_name drop not null;
alter table public.games alter column home_qb_name drop not null;
alter table public.games alter column away_coach drop not null;
alter table public.games alter column home_coach drop not null;
alter table public.games alter column stadium_id drop not null;
alter table public.games alter column stadium drop not null;

-- Backend secret-key clients can access these tables; no anonymous policies
-- are created yet.
alter table public.teams enable row level security;
alter table public.player_external_ids enable row level security;
alter table public.player_season_stats enable row level security;
alter table public.player_weekly_rosters enable row level security;
alter table public.player_snap_counts enable row level security;
alter table public.team_weekly_stats enable row level security;
alter table public.depth_chart_entries enable row level security;
alter table public.current_depth_chart_entries enable row level security;
