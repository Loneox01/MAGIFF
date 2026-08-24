-- Add a points-per-game variant while preserving season totals as the default.

alter table public.roster_game_sessions
    add column scoring_mode text not null default 'season_total';

alter table public.roster_game_sessions
    add constraint roster_game_sessions_scoring_mode_check
    check (scoring_mode in ('season_total', 'ppg'));

create or replace function public.get_roster_game_pool_v2(
    target_season integer,
    target_scoring_mode text default 'season_total'
)
returns table (
    player_id uuid,
    display_name text,
    team text,
    player_position text,
    fantasy_points_ppr double precision,
    team_name text,
    team_logo_url text,
    team_color text
)
language sql
stable
security invoker
set search_path = ''
as $$
    with player_team_totals as (
        select
            stats.player_id,
            stats.team,
            stats.position_group as player_position,
            sum(stats.fantasy_points_ppr)::double precision
                as season_fantasy_points_ppr,
            count(distinct stats.game_id)::double precision as games
        from public.player_weekly_stats as stats
        where stats.season = target_season
          and stats.season_type = 'REG'
          and stats.position_group in ('QB', 'RB', 'WR', 'TE')
        group by stats.player_id, stats.team, stats.position_group
    ),
    scored as (
        select
            totals.*,
            case target_scoring_mode
                when 'ppg' then
                    totals.season_fantasy_points_ppr / nullif(totals.games, 0)
                else totals.season_fantasy_points_ppr
            end as score_value
        from player_team_totals as totals
    ),
    ranked as (
        select
            scored.*,
            row_number() over (
                partition by scored.team, scored.player_position
                order by
                    scored.score_value desc,
                    scored.season_fantasy_points_ppr desc,
                    scored.player_id
            ) as position_rank
        from scored
    )
    select
        ranked.player_id,
        player.display_name,
        ranked.team,
        ranked.player_position,
        ranked.score_value,
        team.team_name,
        coalesce(team.team_logo_squared, team.team_logo_espn),
        team.team_color
    from ranked
    join public.players as player on player.player_id = ranked.player_id
    join public.teams as team on team.team_abbr = ranked.team
    where target_scoring_mode in ('season_total', 'ppg')
      and ranked.position_rank = 1
    order by ranked.team, ranked.player_position;
$$;
