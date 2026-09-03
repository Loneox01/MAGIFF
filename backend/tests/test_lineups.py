import unittest
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from datetime import timedelta

from league_management.models import (
    LeagueContext,
    LeaguePlayer,
    LeagueRoster,
    LineupAssignment,
    ManagedMatchup,
)
from lineups.agent import (
    LineupAgentService,
    LineupRunResult,
    LineupTokenUsage,
    validate_lineup,
)
from lineups.automation import AutomaticLineupReviewService
from lineups.context import LineupContextBuilder
from lineups.models import (
    DecisionConfidence,
    LineupAnalysis,
    LineupCloseCall,
    PreliminaryLineupPlan,
    ProposedStarter,
    RecommendedStarter,
)
from lineups.tools import LineupToolbox
from lineups.reviews import (
    ReviewOutcome,
    health_snapshot,
    kickoff_slates,
    review_is_due,
)
from integrations.sleeper_projections import SleeperWeeklyProjection
from services.news import NewsOutcome, NewsQuery, NewsReport, NewsResult, PlayerCandidate


def _usage(input_tokens=20, output_tokens=5, cached_tokens=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


def _player(sleeper_id, name, position, team):
    return LeaguePlayer(
        sleeper_player_id=sleeper_id,
        player_id=f"internal-{sleeper_id}",
        display_name=name,
        position=position,
        team=team,
        roster_status="ACT",
    )


def _league():
    qb = _player("qb", "Quarterback One", "QB", "BUF")
    injured = _player("hurt", "Injured Runner", "RB", "MIA")
    receiver = _player("wr", "Flex Receiver", "WR", "CIN")
    defense = _player("JAX", "Jacksonville Jaguars D/ST", "DEF", "JAX")
    bench = _player("bench", "Bench Runner", "RB", "NYJ")
    reserve = _player("reserve", "Reserve Runner", "RB", "NE")
    opponent_qb = _player("opp-qb", "Opponent Quarterback", "QB", "KC")
    managed = LeagueRoster(
        roster_id=6,
        owner_id="user-1",
        owner_name="Magiff",
        starters=(
            LineupAssignment("QB", qb),
            LineupAssignment("RB", injured),
            LineupAssignment("FLEX", receiver),
            LineupAssignment("DEF", defense),
        ),
        bench=(bench,),
        reserve=(reserve,),
        taxi=(),
        wins=0,
        losses=0,
        ties=0,
        points_for=0,
        waiver_position=1,
        waiver_budget_used=0,
    )
    opponent = LeagueRoster(
        roster_id=2,
        owner_id="user-2",
        owner_name="Opponent",
        starters=(LineupAssignment("QB", opponent_qb),),
        bench=(),
        reserve=(),
        taxi=(),
        wins=0,
        losses=0,
        ties=0,
        points_for=0,
        waiver_position=2,
        waiver_budget_used=0,
    )
    return LeagueContext(
        league_id="league-1",
        league_name="Lineup Test",
        season=2026,
        status="in_season",
        current_week=1,
        season_type="regular",
        season_start_date="2026-09-09",
        managed_user_id="user-1",
        managed_user_name="Magiff",
        managed_roster_id=6,
        total_rosters=2,
        roster_positions=("QB", "RB", "FLEX", "DEF", "BN", "IR"),
        scoring_settings={"pass_yd": 0.04, "pass_td": 4, "rec": 1},
        waiver_settings={"waiver_budget": 100},
        trade_settings={},
        managed_roster=managed,
        other_rosters=(opponent,),
        matchup=ManagedMatchup(1, 1, 6, 2, "Opponent", 0, 0),
        transactions=(),
        trending_adds=(),
        trending_drops=(),
        available_candidates=(),
        ecr_snapshot_date=None,
        ecr_scoring_format="ppr",
        ecr_league_format="redraft_1qb",
        ecr_source=None,
        ecr_ranking_page=None,
    )


def _projection(
    sleeper_id,
    name,
    position,
    team,
    points,
    injury_status=None,
):
    first, *rest = name.split(" ")
    return SleeperWeeklyProjection(
        sleeper_player_id=sleeper_id,
        display_name=name,
        position=position,
        team=team,
        opponent="TEST",
        season=2026,
        week=1,
        season_type="regular",
        game_date="2026-09-13",
        game_id=f"game-{sleeper_id}",
        updated_at=100,
        company="rotowire",
        stats={"pts_ppr": points},
        game_status="scheduled",
        injury_status=injury_status,
        injury_body_part="Knee" if injury_status else None,
        injury_notes=None,
        injury_start_date="2026-08-30" if injury_status else None,
        news_updated_at=200 if injury_status else None,
    )


class FakeLeagueBuilder:
    def build(self, **_kwargs):
        return _league()


class FakeProjections:
    def weekly_projections(self, **_kwargs):
        return [
            _projection("qb", "Quarterback One", "QB", "BUF", 20),
            _projection("hurt", "Injured Runner", "RB", "MIA", 18, "Out"),
            _projection("wr", "Flex Receiver", "WR", "CIN", 14),
            _projection("JAX", "Jacksonville Jaguars D/ST", "DEF", "JAX", 8),
            _projection("bench", "Bench Runner", "RB", "NYJ", 12),
            _projection("reserve", "Reserve Runner", "RB", "NE", 30),
            _projection("opp-qb", "Opponent Quarterback", "QB", "KC", 19),
        ]


class FakeSchedule:
    def week_games(self, **_kwargs):
        return [
            {
                "game_id": "game-buf",
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "BUF",
                "away_team": "TEST",
            },
            {
                "game_id": "game-mia",
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "MIA",
                "away_team": "NYJ",
            },
            {
                "game_id": "game-cin",
                "gameday": "2026-09-13",
                "gametime": "16:25",
                "home_team": "CIN",
                "away_team": "JAX",
            },
            {
                "game_id": "game-ne",
                "gameday": "2026-09-14",
                "gametime": "20:15",
                "home_team": "NE",
                "away_team": "KC",
            },
        ]


class FakeNews:
    def __init__(self):
        self.queries = []

    def latest(self, query: NewsQuery):
        self.queries.append(query)
        player_name = query.player or f"{query.team} D/ST"
        return NewsResult(
            outcome=NewsOutcome.SUCCESS,
            query=query,
            resolved_player=(
                PlayerCandidate(
                    player_id=f"internal-{player_name}",
                    display_name=player_name,
                    position="RB",
                    team="TEST",
                    status="ACT",
                )
                if query.player
                else None
            ),
            reports=(
                NewsReport(
                    report_id=f"report-{player_name}",
                    title=f"Latest on {player_name}",
                    source="Test",
                    source_url="https://example.com/report",
                    author=None,
                    published_at="2026-09-01T12:00:00+00:00",
                    players=(player_name,),
                    teams=(query.team or "TEST",),
                    document_type="news",
                    storyline=None,
                    content_mode="summary",
                    body="Current lineup-relevant update.",
                ),
            ),
        )


def _context(as_of=None):
    return LineupContextBuilder(
        league_builder=FakeLeagueBuilder(),
        projections=FakeProjections(),
        schedule=FakeSchedule(),
    ).build(
        league_id="league-1",
        user_reference="Magiff",
        as_of=as_of or datetime(2026, 9, 12, 12, tzinfo=UTC),
    )


def _legal_starters(model=ProposedStarter):
    values = [
        ("QB", "qb", "Quarterback One"),
        ("RB", "bench", "Bench Runner"),
        ("FLEX", "wr", "Flex Receiver"),
        ("DEF", "JAX", "Jacksonville Jaguars D/ST"),
    ]
    if model is ProposedStarter:
        return [model(slot_id=slot, sleeper_player_id=pid, player_name=name) for slot, pid, name in values]
    return [
        model(
            slot_id=slot,
            sleeper_player_id=pid,
            player_name=name,
            rationale="Best legal healthy option.",
            confidence=DecisionConfidence.HIGH,
        )
        for slot, pid, name in values
    ]


class LineupContextTests(unittest.TestCase):
    def test_builds_status_aware_projection_baseline(self):
        context = _context()

        self.assertEqual(context.player_by_id["hurt"].injury_code, "O")
        self.assertFalse(context.player_by_id["hurt"].can_enter_lineup)
        self.assertFalse(context.player_by_id["reserve"].can_enter_lineup)
        baseline = {
            row.slot_id: row.sleeper_player_id
            for row in context.projection_baseline
        }
        self.assertEqual(baseline["RB"], "bench")
        self.assertEqual(baseline["FLEX"], "wr")
        self.assertEqual(context.baseline_projected_total, 54)
        self.assertEqual(context.opponent_current_projected_total, 19)
        self.assertFalse(context.lineup_fully_locked)
        self.assertEqual(
            context.player_by_id["wr"].kickoff_at,
            datetime(2026, 9, 13, 20, 25, tzinfo=UTC),
        )

    def test_validation_rejects_unavailable_or_illegal_players(self):
        context = _context()
        invalid = _legal_starters()
        invalid[1] = ProposedStarter(
            slot_id="RB",
            sleeper_player_id="hurt",
            player_name="Injured Runner",
        )

        with self.assertRaisesRegex(RuntimeError, "cannot enter"):
            validate_lineup(context, invalid)

        validated = validate_lineup(context, _legal_starters())
        self.assertEqual(validated.projected_total, 54)
        self.assertEqual(validated.changes[0].incoming_player, "Bench Runner")

    def test_locked_starter_cannot_move_and_locked_bench_cannot_enter(self):
        context = LineupContextBuilder(
            league_builder=FakeLeagueBuilder(),
            projections=FakeProjections(),
            schedule=FakeSchedule(),
        ).build(
            league_id="league-1",
            user_reference="Magiff",
            as_of=datetime(2026, 9, 13, 18, tzinfo=UTC),
        )
        self.assertTrue(context.player_by_id["hurt"].is_locked)
        self.assertTrue(context.player_by_id["bench"].is_locked)

        with self.assertRaisesRegex(RuntimeError, "locked=True"):
            validate_lineup(context, _legal_starters())

        current = [
            ProposedStarter(
                slot_id="QB",
                sleeper_player_id="qb",
                player_name="Quarterback One",
            ),
            ProposedStarter(
                slot_id="RB",
                sleeper_player_id="hurt",
                player_name="Injured Runner",
            ),
            ProposedStarter(
                slot_id="FLEX",
                sleeper_player_id="wr",
                player_name="Flex Receiver",
            ),
            ProposedStarter(
                slot_id="DEF",
                sleeper_player_id="JAX",
                player_name="Jacksonville Jaguars D/ST",
            ),
        ]
        validated = validate_lineup(context, current)
        self.assertEqual(validated.changes, ())


class LineupAgentTests(unittest.TestCase):
    def test_automatically_checks_changed_close_and_injured_players(self):
        preliminary = PreliminaryLineupPlan(
            week=1,
            starters=_legal_starters(),
            news_check_player_ids=["wr"],
            preliminary_strategy="Replace the unavailable runner.",
        )
        final = LineupAnalysis(
            week=1,
            starters=_legal_starters(RecommendedStarter),
            close_calls=[
                LineupCloseCall(
                    selected_player_id="wr",
                    alternative_player_id="bench",
                    rationale="The receiver projects slightly better in FLEX.",
                )
            ],
            overall_strategy="Use the healthy projection baseline.",
            warnings=["Recheck late designations before kickoff."],
        )
        responses = iter(
            [
                SimpleNamespace(output=[], output_parsed=preliminary, usage=_usage(100, 20, 30)),
                SimpleNamespace(output=[], output_parsed=final, usage=_usage(120, 25, 40)),
            ]
        )
        fake_news = FakeNews()
        context = _context()
        service = LineupAgentService(
            client=SimpleNamespace(
                responses=SimpleNamespace(parse=lambda **_kwargs: next(responses))
            )
        )

        result = service.run(
            context,
            toolbox=LineupToolbox(context, news=fake_news),
        )

        queried = [query.player for query in fake_news.queries]
        self.assertCountEqual(
            queried,
            ["Flex Receiver", "Bench Runner", "Injured Runner"],
        )
        self.assertEqual(result.validated.projected_total, 54)
        self.assertEqual(result.usage.input_tokens, 220)
        automatic = [call for call in result.tool_calls if call.automatic]
        self.assertEqual(len(automatic), 3)

    def test_final_lineup_cannot_duplicate_a_player(self):
        context = _context()
        starters = _legal_starters(RecommendedStarter)
        starters[2] = RecommendedStarter(
            slot_id="FLEX",
            sleeper_player_id="bench",
            player_name="Bench Runner",
            rationale="Invalid duplicate.",
            confidence=DecisionConfidence.LOW,
        )
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            validate_lineup(context, starters)


class FakeReviewContextBuilder:
    def build(self, **kwargs):
        return _context(kwargs["as_of"])


class FakeProjectionFailureContextBuilder:
    def build(self, **kwargs):
        return replace(
            _context(kwargs["as_of"]),
            projection_error="projection provider unavailable",
        )


class FakeLineupAgent:
    def __init__(self, *, change=True, error=None):
        self.change = change
        self.error = error
        self.calls = []

    def run(self, context, question):
        self.calls.append((context, question))
        if self.error:
            raise self.error
        starters = _legal_starters() if self.change else [
            ProposedStarter(
                slot_id=slot.slot_id,
                sleeper_player_id=slot.current_player_id,
                player_name=(
                    context.player_by_id[slot.current_player_id].display_name
                    if slot.current_player_id
                    else None
                ),
            )
            for slot in context.slots
        ]
        recommended = [
            RecommendedStarter(
                slot_id=value.slot_id,
                sleeper_player_id=value.sleeper_player_id,
                player_name=value.player_name,
                rationale="Test recommendation.",
                confidence=DecisionConfidence.HIGH,
            )
            for value in starters
        ]
        return LineupRunResult(
            analysis=LineupAnalysis(
                week=context.week,
                starters=recommended,
                close_calls=[],
                overall_strategy="Use the best legal option before this slate.",
                warnings=[],
            ),
            preliminary=PreliminaryLineupPlan(
                week=context.week,
                starters=starters,
                news_check_player_ids=[],
                preliminary_strategy="Test.",
            ),
            validated=validate_lineup(context, starters),
            model="test-model",
            latency_seconds=1,
            tool_rounds=0,
            usage=LineupTokenUsage(10, 2, 3),
            estimated_cost_usd=0.001,
            tool_calls=(),
            news_evidence={},
        )


class FakeReviewRepository:
    def __init__(self, previous=None):
        self.previous = previous
        self.claims = []
        self.finished = []
        self.notifications = []

    def latest_automation_review(self, **_kwargs):
        return self.previous

    def claim(self, **kwargs):
        self.claims.append(kwargs)
        return "review-1"

    def finish(self, review_id, **kwargs):
        self.finished.append((review_id, kwargs))

    def mark_notification(self, review_id, **kwargs):
        self.notifications.append((review_id, kwargs))

    def pending_notifications(self, limit=5):
        return []


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send(self, content):
        self.messages.append(content)
        return "message-1"


class LineupAutomationTests(unittest.TestCase):
    def setUp(self):
        self.as_of = datetime(2026, 9, 13, 15, 50, tzinfo=UTC)

    def test_slate_groups_all_players_at_one_kickoff(self):
        context = _context(self.as_of)
        slate = kickoff_slates(context, as_of=self.as_of)[0]

        self.assertEqual(
            {player.display_name for player in slate.players},
            {"Quarterback One", "Injured Runner", "Bench Runner"},
        )
        self.assertTrue(review_is_due(slate, as_of=self.as_of, lead_minutes=75))
        self.assertFalse(
            review_is_due(
                slate,
                as_of=slate.kickoff_at - timedelta(minutes=76),
                lead_minutes=75,
            )
        )

    def test_forced_e2e_sends_one_pinged_change_message(self):
        agent = FakeLineupAgent(change=True)
        notifier = FakeNotifier()
        service = AutomaticLineupReviewService(
            context_builder=FakeReviewContextBuilder(),
            lineup_agent=agent,
            notifier=notifier,
            owner_discord_user_id="123456",
        )

        result = service.run(
            league_id="league-1",
            user_reference="Magiff",
            as_of=datetime(2026, 9, 12, 12, tzinfo=UTC),
            force_next=True,
            persist=False,
            notify=True,
        )

        self.assertEqual(result.review.outcome, ReviewOutcome.CHANGE_RECOMMENDED)
        self.assertEqual(len(notifier.messages), 1)
        self.assertIn("<@123456>", notifier.messages[0])
        self.assertIn("Injured Runner", notifier.messages[0])
        self.assertIn("Bench Runner", notifier.messages[0])

    def test_no_change_message_does_not_ping(self):
        notifier = FakeNotifier()
        service = AutomaticLineupReviewService(
            context_builder=FakeReviewContextBuilder(),
            lineup_agent=FakeLineupAgent(change=False),
            notifier=notifier,
            owner_discord_user_id="123456",
        )
        result = service.run(
            league_id="league-1",
            user_reference="Magiff",
            as_of=datetime(2026, 9, 13, 18, tzinfo=UTC),
            force_next=True,
            persist=False,
            notify=True,
        )

        self.assertEqual(result.review.outcome, ReviewOutcome.NO_CHANGE)
        self.assertNotIn("<@123456>", notifier.messages[0])

    def test_same_health_snapshot_is_idempotently_skipped(self):
        context = _context(self.as_of)
        repository = FakeReviewRepository(
            previous={"health_snapshot": health_snapshot(context)}
        )
        agent = FakeLineupAgent()
        service = AutomaticLineupReviewService(
            context_builder=FakeReviewContextBuilder(),
            lineup_agent=agent,
            repository=repository,
            owner_discord_user_id="123456",
        )

        result = service.run(
            league_id="league-1",
            user_reference="Magiff",
            as_of=self.as_of,
            persist=True,
            notify=False,
        )

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.reason, "already_reviewed")
        self.assertEqual(agent.calls, [])

    def test_changed_health_snapshot_becomes_emergency_and_pings(self):
        repository = FakeReviewRepository(previous={"health_snapshot": {}})
        notifier = FakeNotifier()
        service = AutomaticLineupReviewService(
            context_builder=FakeReviewContextBuilder(),
            lineup_agent=FakeLineupAgent(change=True),
            repository=repository,
            notifier=notifier,
            owner_discord_user_id="123456",
        )

        result = service.run(
            league_id="league-1",
            user_reference="Magiff",
            as_of=self.as_of,
            persist=True,
            notify=True,
        )

        self.assertEqual(result.review.outcome, ReviewOutcome.EMERGENCY_UPDATE)
        self.assertIn("<@123456>", notifier.messages[0])
        self.assertIn("EMERGENCY UPDATE", notifier.messages[0])

    def test_projection_failure_pings_once_when_review_is_due(self):
        repository = FakeReviewRepository()
        notifier = FakeNotifier()
        service = AutomaticLineupReviewService(
            context_builder=FakeProjectionFailureContextBuilder(),
            lineup_agent=FakeLineupAgent(),
            repository=repository,
            notifier=notifier,
            owner_discord_user_id="123456",
        )

        result = service.run(
            league_id="league-1",
            user_reference="Magiff",
            as_of=self.as_of,
            persist=True,
            notify=True,
        )

        self.assertEqual(result.review.outcome, ReviewOutcome.REVIEW_FAILED)
        self.assertIn("<@123456>", notifier.messages[0])
        self.assertIn("REVIEW FAILED", notifier.messages[0])
        self.assertIn("projections were unavailable", notifier.messages[0])

    def test_projection_failure_is_silent_before_review_window(self):
        notifier = FakeNotifier()
        service = AutomaticLineupReviewService(
            context_builder=FakeProjectionFailureContextBuilder(),
            lineup_agent=FakeLineupAgent(),
            notifier=notifier,
            owner_discord_user_id="123456",
        )

        result = service.run(
            league_id="league-1",
            user_reference="Magiff",
            as_of=datetime(2026, 9, 12, 12, tzinfo=UTC),
            persist=False,
            notify=True,
        )

        self.assertEqual(result.reason, "before_review_window")
        self.assertEqual(notifier.messages, [])


if __name__ == "__main__":
    unittest.main()
