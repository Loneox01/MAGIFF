-- Persistent application identities and the Discord roster-roulette game.
-- The backend secret-key client owns all access; no anonymous policies exist.

create table public.app_users (
    app_user_id uuid primary key,
    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now()
);

create table public.user_identities (
    user_identity_id bigint generated always as identity primary key,
    app_user_id uuid not null references public.app_users(app_user_id) on delete cascade,
    provider text not null,
    provider_user_id text not null,
    display_name text,
    created_at timestamp with time zone not null default now(),
    last_seen_at timestamp with time zone not null default now(),

    constraint user_identities_provider_check
        check (btrim(provider) <> '' and btrim(provider_user_id) <> ''),
    constraint user_identities_provider_user_key
        unique (provider, provider_user_id)
);

create index user_identities_app_user_idx
    on public.user_identities (app_user_id);

create or replace function public.ensure_app_user_identity(
    p_provider text,
    p_provider_user_id text,
    p_display_name text default null
)
returns uuid
language plpgsql
security invoker
set search_path = ''
as $$
declare
    resolved_user_id uuid;
    candidate_user_id uuid := gen_random_uuid();
begin
    select identity.app_user_id
      into resolved_user_id
      from public.user_identities as identity
     where identity.provider = p_provider
       and identity.provider_user_id = p_provider_user_id;

    if resolved_user_id is not null then
        update public.user_identities
           set display_name = coalesce(p_display_name, display_name),
               last_seen_at = now()
         where provider = p_provider
           and provider_user_id = p_provider_user_id;
        return resolved_user_id;
    end if;

    insert into public.app_users (app_user_id)
    values (candidate_user_id);

    begin
        insert into public.user_identities (
            app_user_id,
            provider,
            provider_user_id,
            display_name
        ) values (
            candidate_user_id,
            p_provider,
            p_provider_user_id,
            p_display_name
        );
        return candidate_user_id;
    exception when unique_violation then
        delete from public.app_users
         where app_user_id = candidate_user_id;

        update public.user_identities
           set display_name = coalesce(p_display_name, display_name),
               last_seen_at = now()
         where provider = p_provider
           and provider_user_id = p_provider_user_id
        returning app_user_id into resolved_user_id;

        return resolved_user_id;
    end;
end;
$$;

create table public.roster_game_sessions (
    game_id uuid primary key,
    app_user_id uuid not null references public.app_users(app_user_id) on delete cascade,
    discord_user_id text not null,
    discord_guild_id text not null,
    season integer not null,
    reveal_during_roll boolean not null default false,
    status text not null default 'active',
    pending_pick jsonb,
    team_reroll_used boolean not null default false,
    position_reroll_used boolean not null default false,
    total_points double precision,
    wins integer,
    losses integer,
    version integer not null default 0,
    started_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now(),
    completed_at timestamp with time zone,

    constraint roster_game_sessions_status_check
        check (status in ('active', 'completed', 'abandoned')),
    constraint roster_game_sessions_record_check
        check (
            (wins is null and losses is null)
            or (
                wins between 0 and 17
                and losses between 0 and 17
                and wins + losses = 17
            )
        )
);

create index roster_game_sessions_user_started_idx
    on public.roster_game_sessions (app_user_id, started_at desc);
create index roster_game_sessions_guild_started_idx
    on public.roster_game_sessions (discord_guild_id, started_at desc);

create table public.roster_game_picks (
    game_id uuid not null references public.roster_game_sessions(game_id) on delete cascade,
    pick_number integer not null,
    roster_slot text not null,
    player_id uuid not null references public.players(player_id),
    display_name text not null,
    team text not null,
    position text not null,
    fantasy_points_ppr double precision not null,
    team_name text not null,
    team_logo_url text,
    team_color text,
    created_at timestamp with time zone not null default now(),

    primary key (game_id, pick_number),
    constraint roster_game_picks_slot_key unique (game_id, roster_slot),
    constraint roster_game_picks_team_key unique (game_id, team),
    constraint roster_game_picks_player_key unique (game_id, player_id),
    constraint roster_game_picks_number_check check (pick_number between 1 and 7),
    constraint roster_game_picks_slot_check
        check (roster_slot in ('QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX')),
    constraint roster_game_picks_position_check
        check (position in ('QB', 'RB', 'WR', 'TE'))
);

create table public.roster_game_actions (
    interaction_id text primary key,
    game_id uuid not null references public.roster_game_sessions(game_id) on delete cascade,
    app_user_id uuid not null references public.app_users(app_user_id) on delete cascade,
    action text not null,
    resulting_version integer not null,
    created_at timestamp with time zone not null default now(),

    constraint roster_game_actions_action_check
        check (action in ('lock', 'reroll_team', 'reroll_position'))
);

create index roster_game_actions_game_idx
    on public.roster_game_actions (game_id, created_at);

-- One row per team and offensive fantasy position. Traded players are scored
-- only for the team-specific weekly rows they actually accumulated there.
create or replace function public.get_roster_game_pool(target_season integer)
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
            sum(stats.fantasy_points_ppr) as fantasy_points_ppr
        from public.player_weekly_stats as stats
        where stats.season = target_season
          and stats.season_type = 'REG'
          and stats.position_group in ('QB', 'RB', 'WR', 'TE')
        group by stats.player_id, stats.team, stats.position_group
    ),
    ranked as (
        select
            totals.*,
            row_number() over (
                partition by totals.team, totals.player_position
                order by totals.fantasy_points_ppr desc, totals.player_id
            ) as position_rank
        from player_team_totals as totals
    )
    select
        ranked.player_id,
        player.display_name,
        ranked.team,
        ranked.player_position,
        ranked.fantasy_points_ppr,
        team.team_name,
        coalesce(team.team_logo_squared, team.team_logo_espn),
        team.team_color
    from ranked
    join public.players as player on player.player_id = ranked.player_id
    join public.teams as team on team.team_abbr = ranked.team
    where ranked.position_rank = 1
    order by ranked.team, ranked.player_position;
$$;

-- Apply one button action atomically. The version and interaction ID make
-- stale clicks and Discord retries harmless, while table constraints enforce
-- the no-repeat slot, team, and player rules independently of application code.
create or replace function public.apply_roster_game_transition(
    p_game_id uuid,
    p_app_user_id uuid,
    p_expected_version integer,
    p_interaction_id text,
    p_action text,
    p_pending_pick jsonb,
    p_team_reroll_used boolean,
    p_position_reroll_used boolean,
    p_status text,
    p_total_points double precision,
    p_wins integer,
    p_losses integer,
    p_new_pick jsonb default null
)
returns boolean
language plpgsql
security invoker
set search_path = ''
as $$
declare
    current_version integer;
begin
    select session.version
      into current_version
      from public.roster_game_sessions as session
     where session.game_id = p_game_id
       and session.app_user_id = p_app_user_id
     for update;

    if current_version is null or current_version <> p_expected_version then
        return false;
    end if;

    if exists (
        select 1
          from public.roster_game_actions as action
         where action.interaction_id = p_interaction_id
    ) then
        return false;
    end if;

    if p_new_pick is not null then
        insert into public.roster_game_picks (
            game_id,
            pick_number,
            roster_slot,
            player_id,
            display_name,
            team,
            position,
            fantasy_points_ppr,
            team_name,
            team_logo_url,
            team_color
        ) values (
            p_game_id,
            (p_new_pick->>'pick_number')::integer,
            p_new_pick->>'roster_slot',
            (p_new_pick->>'player_id')::uuid,
            p_new_pick->>'display_name',
            p_new_pick->>'team',
            p_new_pick->>'position',
            (p_new_pick->>'fantasy_points_ppr')::double precision,
            p_new_pick->>'team_name',
            p_new_pick->>'team_logo_url',
            p_new_pick->>'team_color'
        );
    end if;

    update public.roster_game_sessions
       set pending_pick = p_pending_pick,
           team_reroll_used = p_team_reroll_used,
           position_reroll_used = p_position_reroll_used,
           status = p_status,
           total_points = p_total_points,
           wins = p_wins,
           losses = p_losses,
           version = p_expected_version + 1,
           updated_at = now(),
           completed_at = case
               when p_status = 'completed' then now()
               else completed_at
           end
     where game_id = p_game_id;

    insert into public.roster_game_actions (
        interaction_id,
        game_id,
        app_user_id,
        action,
        resulting_version
    ) values (
        p_interaction_id,
        p_game_id,
        p_app_user_id,
        p_action,
        p_expected_version + 1
    );

    return true;
end;
$$;

alter table public.app_users enable row level security;
alter table public.user_identities enable row level security;
alter table public.roster_game_sessions enable row level security;
alter table public.roster_game_picks enable row level security;
alter table public.roster_game_actions enable row level security;
