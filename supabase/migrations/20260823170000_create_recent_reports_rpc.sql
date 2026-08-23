-- Deterministic recency retrieval for non-agent clients such as Discord /news.
-- This intentionally bypasses embeddings, hybrid search, planning, and reranking.

create or replace function public.get_recent_reports(
    match_count integer default 5,
    filter_player_id uuid default null,
    filter_team text default null
)
returns table (
    report_id text,
    title text,
    source text,
    source_url text,
    author text,
    published_at timestamp with time zone,
    player_ids uuid[],
    player_names text[],
    teams text[],
    document_type text,
    storyline text,
    content_mode text,
    body text
)
language sql
stable
security invoker
set search_path = ''
as $$
    select
        report.report_id,
        report.title,
        report.source,
        report.source_url,
        report.author,
        report.published_at,
        array(
            select report_player.player_id
            from public.report_players as report_player
            where report_player.report_id = report.report_id
            order by
                case report_player.mention_role
                    when 'primary_subject' then 0
                    when 'materially_affected' then 1
                    else 2
                end,
                report_player.player_id
        ),
        array(
            select player.display_name
            from public.report_players as report_player
            join public.players as player
              on player.player_id = report_player.player_id
            where report_player.report_id = report.report_id
            order by
                case report_player.mention_role
                    when 'primary_subject' then 0
                    when 'materially_affected' then 1
                    else 2
                end,
                report_player.player_id
        ),
        report.teams,
        report.document_type,
        report.storyline,
        report.content_mode,
        report.body
    from public.reports as report
    where report.is_active
      and (
          filter_player_id is null
          or exists (
              select 1
              from public.report_players as report_player
              where report_player.report_id = report.report_id
                and report_player.player_id = filter_player_id
          )
      )
      and (filter_team is null or report.teams @> array[filter_team])
    order by report.published_at desc, report.report_id
    limit least(greatest(match_count, 1), 10);
$$;

revoke execute on function public.get_recent_reports(integer, uuid, text)
from public, anon, authenticated;

grant execute on function public.get_recent_reports(integer, uuid, text)
to service_role;
