-- The 17-0 Challenge originally shipped before the forfeit action existed.
-- Replace the deployed constraint without disturbing existing game actions.

alter table public.roster_game_actions
    drop constraint if exists roster_game_actions_action_check;

alter table public.roster_game_actions
    add constraint roster_game_actions_action_check
    check (action in ('lock', 'reroll_team', 'reroll_position', 'forfeit'));
