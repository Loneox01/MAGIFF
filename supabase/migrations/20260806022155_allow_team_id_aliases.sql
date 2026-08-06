-- A franchise team_id is intentionally shared by historical abbreviations,
-- such as LA/LAR/STL and LV/OAK. The abbreviation identifies each row.
alter table public.teams drop constraint teams_team_id_key;

create index teams_team_id_idx on public.teams (team_id);
