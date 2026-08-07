-- Allow ranking formats whose upstream page does not state a scoring system,
-- distinguish non-redraft season-opening snapshots, and preserve the exact
-- source ranking page used during processing.

alter table public.player_ecr
    drop constraint player_ecr_scoring_format_check;

alter table public.player_ecr
    add constraint player_ecr_scoring_format_check
    check (scoring_format in ('ppr', 'half_ppr', 'standard', 'source_default'));

alter table public.player_ecr
    drop constraint player_ecr_snapshot_type_check;

alter table public.player_ecr
    add constraint player_ecr_snapshot_type_check
    check (snapshot_type in ('current', 'final_preseason', 'season_opening'));

alter table public.player_ecr
    add column ranking_page text;
