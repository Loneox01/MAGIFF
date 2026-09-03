import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

import httpx

from integrations.fantasycalc import FantasyCalcClient, FantasyCalcValue
from integrations.sleeper_projections import (
    SleeperProjectionClient,
    SleeperWeeklyProjection,
)
from league_management.models import (
    AvailableCandidate,
    LeagueContext,
    LeaguePlayer,
    LeagueRoster,
    LineupAssignment,
)
from services.news import (
    NewsOutcome,
    NewsQuery,
    NewsReport,
    NewsResult,
    PlayerCandidate,
)
from waivers.agent import WaiverAgentService
from waivers.context import WaiverContextBuilder
from waivers.models import (
    CandidateRole,
    PreliminaryWaiverAnalysis,
    PreliminaryWaiverMove,
    RecommendationPriority,
    TeamNeed,
    TimeHorizon,
    WaiverAction,
    WaiverAnalysis,
    WaiverCandidate,
    WaiverContext,
    WaiverRecommendation,
)
from waivers.tools import WaiverToolbox


def _usage(input_tokens=20, output_tokens=5, cached_tokens=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


def _candidate(
    sleeper_id: str,
    name: str,
    position: str,
    team: str,
    value: int,
    trend: int,
    ecr: float | None,
    projected_points: float | None = None,
    opponent: str | None = None,
) -> WaiverCandidate:
    return WaiverCandidate(
        sleeper_player_id=sleeper_id,
        player_id=f"internal-{sleeper_id}",
        display_name=name,
        position=position,
        team=team,
        roster_status="ACT",
        fantasycalc_value=value,
        fantasycalc_overall_rank=max(1, 10000 - value),
        fantasycalc_position_rank=1,
        fantasycalc_trend_30_day=trend,
        roster_percent=0.5,
        trade_frequency=0.01,
        ecr=ecr,
        ecr_position_rank=1 if ecr is not None else None,
        projection_week=1 if projected_points is not None else None,
        projected_points=projected_points,
        projection_opponent=opponent,
        projection_game_date="2026-09-13" if projected_points is not None else None,
        projection_updated_at=1 if projected_points is not None else None,
        projection_source="rotowire" if projected_points is not None else None,
    )


def _waiver_context() -> WaiverContext:
    starter = LeaguePlayer(
        sleeper_player_id="starter",
        player_id="internal-starter",
        display_name="Starting Player",
        position="RB",
        team="BUF",
        roster_status="ACT",
    )
    bench = LeaguePlayer(
        sleeper_player_id="bench",
        player_id="internal-bench",
        display_name="Bench Player",
        position="WR",
        team="CHI",
        roster_status="ACT",
    )
    roster = LeagueRoster(
        roster_id=6,
        owner_id="user-1",
        owner_name="Magiff",
        starters=(LineupAssignment(slot="RB", player=starter),),
        bench=(bench,),
        reserve=(),
        taxi=(),
        wins=0,
        losses=0,
        ties=0,
        points_for=0,
        waiver_position=2,
        waiver_budget_used=0,
    )
    league = LeagueContext(
        league_id="league-1",
        league_name="Test League",
        season=2026,
        status="in_season",
        current_week=1,
        season_type="regular",
        season_start_date="2026-09-09",
        managed_user_id="user-1",
        managed_user_name="Magiff",
        managed_roster_id=6,
        total_rosters=8,
        roster_positions=("QB", "RB", "WR", "FLEX", "BN"),
        scoring_settings={"rec": 1.0},
        waiver_settings={"waiver_budget": 100},
        trade_settings={},
        managed_roster=roster,
        other_rosters=(),
        matchup=None,
        transactions=(),
        trending_adds=(),
        trending_drops=(),
        available_candidates=(
            AvailableCandidate(
                player_id="internal-add",
                sleeper_player_id="add",
                display_name="Available Player",
                position="RB",
                team="BUF",
                overall_rank=80,
                position_rank=30,
            ),
        ),
        ecr_snapshot_date="2026-08-28",
        ecr_scoring_format="ppr",
        ecr_league_format="redraft_1qb",
        ecr_source="FantasyPros",
        ecr_ranking_page="consensus-cheatsheets",
    )
    return WaiverContext(
        league=league,
        available_players=(
            _candidate(
                "add", "Available Player", "RB", "BUF", 5000, 400, 80, 15.5, "MIA"
            ),
            _candidate(
                "stash", "Deep Stash", "WR", "BUF", 3000, 800, 130, 10.0, "MIA"
            ),
            _candidate(
                "other", "Other Player", "TE", "NYJ", 3500, 20, None, 7.5, "NE"
            ),
        ),
        managed_players=(
            _candidate(
                "bench", "Bench Player", "WR", "CHI", 2500, -50, 140, 8.0, "GB"
            ),
        ),
        top_default_count=2,
    )


class FakeNewsService:
    def __init__(self):
        self.queries: list[NewsQuery] = []

    def latest(self, query: NewsQuery) -> NewsResult:
        self.queries.append(query)
        player = PlayerCandidate(
            player_id=f"internal-{query.player}",
            display_name=query.player or "Unknown",
            position="RB",
            team="BUF",
            status="ACT",
        )
        return NewsResult(
            outcome=NewsOutcome.SUCCESS,
            query=query,
            resolved_player=player,
            reports=(
                NewsReport(
                    report_id=f"report-{query.player}",
                    title=f"Latest on {query.player}",
                    source="Test Source",
                    source_url="https://example.com/report",
                    author=None,
                    published_at="2026-08-31T12:00:00+00:00",
                    players=(query.player or "",),
                    teams=("BUF",),
                    document_type="news",
                    storyline=None,
                    content_mode="summary",
                    body="Current role update.",
                ),
            ),
        )


class FakePlayerSearch:
    def __init__(self, named=None, grouped=None):
        self.named = named or []
        self.grouped = grouped or []

    def find_players(self, _name):
        return list(self.named)

    def team_position_players(self, **_kwargs):
        return list(self.grouped)


class FakeLeagueBuilder:
    def __init__(self, league):
        self.league = league
        self.calls = []

    def build(self, **kwargs):
        self.calls.append(kwargs)
        return self.league


class FakeIdentityRepository:
    def resolve_players(self, sleeper_ids):
        return {
            sleeper_id: {
                "player_id": f"internal-{sleeper_id}",
                "display_name": {
                    "add": "Available Player",
                    "bench": "Bench Player",
                }.get(sleeper_id, sleeper_id),
                "position": "RB" if sleeper_id == "add" else "WR",
                "team": "BUF" if sleeper_id == "add" else "CHI",
                "roster_status": "ACT",
            }
            for sleeper_id in sleeper_ids
        }


class FakeFantasyCalc:
    def current_redraft_values(self, **_kwargs):
        return [
            FantasyCalcValue(
                fantasycalc_player_id="1",
                sleeper_player_id="add",
                display_name="Available Player",
                position="RB",
                team="BUF",
                value=5000,
                overall_rank=50,
                position_rank=20,
                trend_30_day=200,
                roster_percent=0.4,
                trade_frequency=0.01,
            ),
            FantasyCalcValue(
                fantasycalc_player_id="2",
                sleeper_player_id="bench",
                display_name="Bench Player",
                position="WR",
                team="CHI",
                value=2500,
                overall_rank=100,
                position_rank=40,
                trend_30_day=-50,
                roster_percent=0.3,
                trade_frequency=0.01,
            ),
        ]


def _projection(
    sleeper_id: str,
    name: str,
    position: str,
    team: str,
    opponent: str,
    week: int,
    points: float,
) -> SleeperWeeklyProjection:
    return SleeperWeeklyProjection(
        sleeper_player_id=sleeper_id,
        display_name=name,
        position=position,
        team=team,
        opponent=opponent,
        season=2026,
        week=week,
        season_type="regular",
        game_date=f"2026-09-{12 + week:02d}",
        game_id=f"game-{week}-{sleeper_id}",
        updated_at=1000 + week,
        company="rotowire",
        stats={"custom_projection": points, "pts_ppr": points},
    )


class FakeProjections:
    def __init__(self, by_week=None):
        self.by_week = by_week or {
            1: [
                _projection(
                    "add", "Available Player", "RB", "BUF", "MIA", 1, 15.5
                )
            ]
        }
        self.calls = []

    def weekly_projections(self, **kwargs):
        self.calls.append(kwargs)
        positions = set(kwargs.get("positions") or [])
        return [
            item
            for item in self.by_week.get(kwargs["week"], [])
            if not positions or item.position in positions
        ]

class FantasyCalcClientTests(unittest.TestCase):
    def test_normalizes_live_payload_shape(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.params["numTeams"], "8")
            self.assertEqual(request.url.params["ppr"], "1.0")
            return httpx.Response(
                200,
                json=[
                    {
                        "player": {
                            "id": 1,
                            "name": "Available Player",
                            "sleeperId": "add",
                            "position": "RB",
                            "maybeTeam": "BUF",
                        },
                        "value": 5000,
                        "overallRank": 50,
                        "positionRank": 20,
                        "trend30Day": 200,
                        "maybeRosterPercent": 0.4,
                        "maybeTradeFrequency": 0.01,
                    }
                ],
            )

        values = FantasyCalcClient(
            client=httpx.Client(transport=httpx.MockTransport(handler))
        ).current_redraft_values(teams=8, quarterback_slots=1, ppr=1.0)

        self.assertEqual(values[0].sleeper_player_id, "add")
        self.assertEqual(values[0].trend_30_day, 200)


class SleeperProjectionClientTests(unittest.TestCase):
    def test_normalizes_projection_and_uses_league_scoring(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.params["season_type"], "regular")
            self.assertEqual(request.url.params.get_list("position[]"), ["RB"])
            return httpx.Response(
                200,
                json=[
                    {
                        "season": "2026",
                        "week": 1,
                        "season_type": "regular",
                        "date": "2026-09-13",
                        "player_id": "add",
                        "team": "BUF",
                        "opponent": "MIA",
                        "updated_at": 1234,
                        "company": "rotowire",
                        "player": {
                            "first_name": "Available",
                            "last_name": "Player",
                            "position": "RB",
                            "fantasy_positions": ["RB"],
                            "injury_status": "Questionable",
                            "injury_body_part": "Hamstring",
                            "injury_notes": "Limited",
                            "injury_start_date": "2026-08-30",
                            "news_updated": 1235,
                        },
                        "stats": {
                            "rush_yd": 100,
                            "rush_td": 1,
                            "rec": 2,
                            "pts_ppr": 20,
                        },
                    }
                ],
            )

        rows = SleeperProjectionClient(
            client=httpx.Client(transport=httpx.MockTransport(handler))
        ).weekly_projections(
            season=2026,
            week=1,
            positions=("RB",),
        )

        self.assertEqual(rows[0].display_name, "Available Player")
        self.assertEqual(rows[0].injury_status, "Questionable")
        self.assertEqual(rows[0].injury_body_part, "Hamstring")
        self.assertEqual(rows[0].news_updated_at, 1235)
        self.assertEqual(
            rows[0].projected_points(
                {"rush_yd": 0.1, "rush_td": 6, "rec": 1}
            ),
            18.0,
        )


class WaiverToolboxTests(unittest.TestCase):
    def test_context_builder_filters_rostered_market_values(self):
        league = _waiver_context().league
        builder = WaiverContextBuilder(
            league_builder=FakeLeagueBuilder(league),
            players=FakeIdentityRepository(),
            fantasycalc=FakeFantasyCalc(),
            projections=FakeProjections(),
        )

        context = builder.build(
            league_id="league-1",
            user_reference="Magiff",
            top_default_count=5,
        )

        self.assertEqual(
            [item.display_name for item in context.available_players],
            ["Available Player"],
        )
        self.assertEqual(
            [item.display_name for item in context.managed_players],
            ["Starting Player", "Bench Player"],
        )
        self.assertEqual(context.available_players[0].projected_points, 15.5)

    def test_filters_market_pool_without_exposing_every_player(self):
        toolbox = WaiverToolbox(
            _waiver_context(),
            news=FakeNewsService(),
            season_stats=lambda *_args: {"fantasy_points_ppr": 100},
            depth_chart=lambda *_args: [],
            player_search=FakePlayerSearch(),
        )
        result = toolbox.rank_available_players(
            position="RB",
            team="BUF",
            sort_by="fantasycalc_trend_30_day",
            limit=5,
        )

        self.assertEqual(result["returned"], 1)
        self.assertEqual(result["players"][0]["name"], "Available Player")
        self.assertEqual(
            toolbox.get_available_player("Deep Stash")["status"],
            "available",
        )

        projected = toolbox.rank_available_players(
            position=None,
            team=None,
            sort_by="sleeper_projection",
            limit=2,
        )
        self.assertEqual(projected["players"][0]["name"], "Available Player")
        self.assertEqual(
            projected["players"][0]["weekly_projection"]["points"],
            15.5,
        )

    def test_named_search_can_reach_available_player_outside_market_pool(self):
        deep_profile = {
            "player_id": "internal-deeper",
            "sleeper_player_id": "deeper",
            "display_name": "Deeper Sleeper",
            "position": "RB",
            "team": "BUF",
            "roster_status": "ACT",
        }
        toolbox = WaiverToolbox(
            _waiver_context(),
            news=FakeNewsService(),
            season_stats=lambda *_args: {"fantasy_points_ppr": 20},
            depth_chart=lambda *_args: [],
            player_search=FakePlayerSearch(named=[deep_profile]),
        )

        result = toolbox.get_available_player("Deeper Sleeper")

        self.assertEqual(result["status"], "available")
        self.assertIsNone(result["player"]["fantasycalc"]["value"])

    def test_streaming_defenses_excludes_rostered_teams_and_compares_current(self):
        base = _waiver_context()
        managed_defense = LeaguePlayer(
            sleeper_player_id="JAX",
            player_id=None,
            display_name="Jacksonville Jaguars D/ST",
            position="DEF",
            team="JAX",
            roster_status="ACT",
        )
        other_defense = LeaguePlayer(
            sleeper_player_id="SEA",
            player_id=None,
            display_name="Seattle Seahawks D/ST",
            position="DEF",
            team="SEA",
            roster_status="ACT",
        )
        managed_roster = replace(
            base.league.managed_roster,
            starters=(
                *base.league.managed_roster.starters,
                LineupAssignment(slot="DEF", player=managed_defense),
            ),
        )
        other_roster = LeagueRoster(
            roster_id=7,
            owner_id="user-2",
            owner_name="Opponent",
            starters=(LineupAssignment(slot="DEF", player=other_defense),),
            bench=(),
            reserve=(),
            taxi=(),
            wins=0,
            losses=0,
            ties=0,
            points_for=0,
            waiver_position=3,
            waiver_budget_used=0,
        )
        league = replace(
            base.league,
            managed_roster=managed_roster,
            other_rosters=(other_roster,),
        )
        context = replace(
            base,
            league=league,
            available_players=(
                *base.available_players,
                _candidate(
                    "TEN",
                    "Tennessee Titans D/ST",
                    "DEF",
                    "TEN",
                    0,
                    0,
                    None,
                    9.0,
                    "NYJ",
                ),
            ),
            managed_players=(
                *base.managed_players,
                _candidate(
                    "JAX",
                    "Jacksonville Jaguars D/ST",
                    "DEF",
                    "JAX",
                    0,
                    0,
                    None,
                    6.0,
                    "CLE",
                ),
            ),
        )
        projections = FakeProjections(
            {
                2: [
                    _projection(
                        "JAX", "Jacksonville Jaguars D/ST", "DEF", "JAX", "KC", 2, 7.0
                    ),
                    _projection(
                        "SEA", "Seattle Seahawks D/ST", "DEF", "SEA", "SF", 2, 10.0
                    ),
                    _projection(
                        "TEN", "Tennessee Titans D/ST", "DEF", "TEN", "IND", 2, 8.0
                    ),
                ]
            }
        )
        toolbox = WaiverToolbox(
            context,
            news=FakeNewsService(),
            player_search=FakePlayerSearch(),
            projections=projections,
        )

        result = toolbox.rank_streaming_defenses(
            week=1,
            lookahead_weeks=2,
            limit=5,
        )

        self.assertEqual(result["available_defenses"][0]["team"], "TEN")
        self.assertEqual(
            result["available_defenses"][0]["current_week_advantage"],
            3.0,
        )
        self.assertNotIn(
            "SEA",
            [item["team"] for item in result["available_defenses"]],
        )
        self.assertEqual(result["current_defenses"][0]["team"], "JAX")

    def test_defense_news_verification_uses_team_scope(self):
        context = replace(
            _waiver_context(),
            available_players=(
                *_waiver_context().available_players,
                _candidate(
                    "TEN",
                    "Tennessee Titans D/ST",
                    "DEF",
                    "TEN",
                    0,
                    0,
                    None,
                    9.0,
                    "NYJ",
                ),
            ),
        )
        toolbox = WaiverToolbox(
            context,
            news=FakeNewsService(),
            player_search=FakePlayerSearch(),
        )

        self.assertEqual(
            toolbox.news_arguments_for("Tennessee Titans D/ST"),
            {"player_ref": None, "team": "TEN", "limit": 3},
        )


class WaiverAgentTests(unittest.TestCase):
    def test_automatically_news_checks_every_shortlisted_add_and_drop(self):
        tool_call = SimpleNamespace(
            type="function_call",
            name="rank_available_players",
            call_id="call-1",
            arguments=json.dumps(
                {
                    "position": "RB",
                    "team": None,
                    "sort_by": "fantasycalc_value",
                    "limit": 5,
                }
            ),
        )
        preliminary = PreliminaryWaiverAnalysis(
            team_needs=[TeamNeed.UPSIDE],
            shortlist=[
                PreliminaryWaiverMove(
                    add_player="Available Player",
                    drop_player="Bench Player",
                    candidate_role=CandidateRole.UPSIDE_STASH,
                    time_horizon=TimeHorizon.LONG_TERM,
                    rationale="Potential improvement over the final bench spot.",
                )
            ],
            preliminary_strategy="Investigate one upside swap.",
            no_action_is_plausible=True,
        )
        final = WaiverAnalysis(
            team_needs=[TeamNeed.UPSIDE],
            recommendations=[
                WaiverRecommendation(
                    action=WaiverAction.SUBMIT_CLAIM,
                    add_player="Available Player",
                    drop_player="Bench Player",
                    candidate_role=CandidateRole.UPSIDE_STASH,
                    priority=RecommendationPriority.LOW,
                    time_horizon=TimeHorizon.LONG_TERM,
                    faab_bid=2,
                    immediate_lineup_impact="None expected.",
                    long_term_value="Adds contingent upside.",
                    add_over_drop="The add has the stronger ceiling.",
                    evidence_summary="Both players received current news checks.",
                    risks=["Role remains uncertain."],
                    confidence=0.65,
                )
            ],
            overall_strategy="Use a small bid for upside.",
            no_action_reason=None,
        )
        responses = iter(
            [
                SimpleNamespace(
                    output=[tool_call],
                    output_parsed=None,
                    usage=_usage(100, 10, 20),
                ),
                SimpleNamespace(
                    output=[],
                    output_parsed=preliminary,
                    usage=_usage(150, 25, 50),
                ),
                SimpleNamespace(
                    output=[],
                    output_parsed=final,
                    usage=_usage(200, 30, 80),
                ),
            ]
        )
        requests = []

        def parse(**kwargs):
            requests.append(kwargs)
            return next(responses)

        news = FakeNewsService()
        context = _waiver_context()
        toolbox = WaiverToolbox(
            context,
            news=news,
            season_stats=lambda *_args: {},
            depth_chart=lambda *_args: [],
            player_search=FakePlayerSearch(),
        )
        service = WaiverAgentService(
            client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
            model="gpt-5.6-terra",
        )

        result = service.run(context, toolbox=toolbox)

        self.assertEqual(result.analysis.recommendations[0].faab_bid, 2)
        self.assertEqual(
            [query.player for query in news.queries],
            ["Available Player", "Bench Player"],
        )
        automatic = [call for call in result.tool_calls if call.automatic]
        self.assertEqual(len(automatic), 2)
        self.assertEqual(requests[-1]["text_format"], WaiverAnalysis)
        self.assertEqual(result.usage.input_tokens, 450)
        self.assertEqual(result.usage.cached_input_tokens, 150)

    def test_final_cannot_introduce_unverified_transaction_player(self):
        preliminary = PreliminaryWaiverAnalysis(
            team_needs=[TeamNeed.NO_CLEAR_NEED],
            shortlist=[],
            preliminary_strategy="Hold.",
            no_action_is_plausible=True,
        )
        invalid_final = WaiverAnalysis(
            team_needs=[TeamNeed.UPSIDE],
            recommendations=[
                WaiverRecommendation(
                    action=WaiverAction.WATCH,
                    add_player="Unverified Player",
                    drop_player=None,
                    candidate_role=CandidateRole.SPECULATIVE_ADD,
                    priority=RecommendationPriority.LOW,
                    time_horizon=TimeHorizon.LONG_TERM,
                    faab_bid=None,
                    immediate_lineup_impact="None.",
                    long_term_value="Unknown.",
                    add_over_drop="No drop proposed.",
                    evidence_summary="None.",
                    risks=["Unverified."],
                    confidence=0.1,
                )
            ],
            overall_strategy="Watch.",
            no_action_reason=None,
        )
        responses = iter(
            [
                SimpleNamespace(
                    output=[], output_parsed=preliminary, usage=_usage()
                ),
                SimpleNamespace(
                    output=[], output_parsed=invalid_final, usage=_usage()
                ),
            ]
        )
        service = WaiverAgentService(
            client=SimpleNamespace(
                responses=SimpleNamespace(parse=lambda **_kwargs: next(responses))
            )
        )

        with self.assertRaisesRegex(RuntimeError, "bypassed"):
            service.run(_waiver_context(), toolbox=WaiverToolbox(
                _waiver_context(),
                news=FakeNewsService(),
                season_stats=lambda *_args: {},
                depth_chart=lambda *_args: [],
                player_search=FakePlayerSearch(),
            ))


if __name__ == "__main__":
    unittest.main()
