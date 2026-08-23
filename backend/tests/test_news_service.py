import unittest

from services.news import (
    MAX_FULL_NEWS_COUNT,
    NewsDetail,
    NewsOutcome,
    NewsQuery,
    NewsReport,
    NewsService,
    PlayerCandidate,
    TeamCandidate,
)


TEAMS = [
    TeamCandidate("ATL", "Atlanta Falcons", "Falcons"),
    TeamCandidate("LA", "Los Angeles Rams", "Rams"),
    TeamCandidate("LAC", "Los Angeles Chargers", "Chargers"),
    TeamCandidate("LV", "Las Vegas Raiders", "Raiders"),
    TeamCandidate("OAK", "Oakland Raiders", "Raiders"),
    TeamCandidate("MIN", "Minnesota Vikings", "Vikings"),
    TeamCandidate("TB", "Tampa Bay Buccaneers", "Buccaneers"),
]


def make_report(report_id: str = "report-1") -> NewsReport:
    return NewsReport(
        report_id=report_id,
        title="A useful report",
        source="FantasyPros",
        source_url=f"https://example.com/{report_id}",
        author=None,
        published_at="2026-08-23T12:00:00+00:00",
        players=("Kenny Gainwell",),
        teams=("TB",),
        document_type="role_update",
        storyline=None,
        content_mode="provider_news",
        body="Stored provider summary.",
    )


class FakeNewsRepository:
    def __init__(self) -> None:
        self.players: dict[str, list[PlayerCandidate]] = {}
        self.teams = list(TEAMS)
        self.reports = [make_report()]
        self.report_calls: list[dict[str, object]] = []

    def find_players(self, name: str) -> list[PlayerCandidate]:
        return self.players.get(name.casefold(), [])

    def list_teams(self) -> list[TeamCandidate]:
        return self.teams

    def recent_reports(
        self,
        *,
        count: int,
        player_id: str | None,
        team: str | None,
    ) -> list[NewsReport]:
        self.report_calls.append(
            {"count": count, "player_id": player_id, "team": team}
        )
        return self.reports[:count]


class NewsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = FakeNewsRepository()
        self.service = NewsService(self.repository)
        self.gainwell = PlayerCandidate(
            "gainwell-id", "Kenny Gainwell", "RB", "TB", "ACT"
        )

    def test_returns_recent_reports_without_filters(self) -> None:
        result = self.service.latest(NewsQuery(count=7))

        self.assertEqual(result.outcome, NewsOutcome.SUCCESS)
        self.assertEqual(len(result.reports), 1)
        self.assertEqual(
            self.repository.report_calls,
            [{"count": 7, "player_id": None, "team": None}],
        )

    def test_falls_back_to_surname_and_reports_interpretation(self) -> None:
        self.repository.players["gainwell"] = [self.gainwell]

        result = self.service.latest(NewsQuery(player="Kenneth Gainwell"))

        self.assertEqual(result.outcome, NewsOutcome.SUCCESS)
        self.assertEqual(result.resolved_player, self.gainwell)
        self.assertIn("Kenny Gainwell", result.resolution_note or "")
        self.assertEqual(
            self.repository.report_calls[0]["player_id"], "gainwell-id"
        )

    def test_returns_ambiguous_players_without_guessing(self) -> None:
        self.repository.players["justin jefferson"] = [
            PlayerCandidate("one", "Justin Jefferson", "WR", "MIN", "ACT"),
            PlayerCandidate("two", "Justin Jefferson", "DB", "ATL", "ACT"),
        ]

        result = self.service.latest(NewsQuery(player="Justin Jefferson"))

        self.assertEqual(result.outcome, NewsOutcome.PLAYER_AMBIGUOUS)
        self.assertEqual(len(result.player_candidates), 2)
        self.assertEqual(self.repository.report_calls, [])

    def test_team_filter_disambiguates_duplicate_player_names(self) -> None:
        wanted = PlayerCandidate(
            "one", "Justin Jefferson", "WR", "MIN", "ACT"
        )
        self.repository.players["justin jefferson"] = [
            wanted,
            PlayerCandidate("two", "Justin Jefferson", "DB", "ATL", "ACT"),
        ]

        result = self.service.latest(
            NewsQuery(player="Justin Jefferson", team="Vikings")
        )

        self.assertEqual(result.outcome, NewsOutcome.SUCCESS)
        self.assertEqual(result.resolved_player, wanted)
        self.assertEqual(result.resolved_team.code, "MIN")
        self.assertEqual(self.repository.report_calls[0]["team"], "MIN")

    def test_unknown_player_stops_before_report_query(self) -> None:
        result = self.service.latest(NewsQuery(player="Imaginary Player"))

        self.assertEqual(result.outcome, NewsOutcome.PLAYER_NOT_FOUND)
        self.assertEqual(self.repository.report_calls, [])

    def test_resolves_team_code_official_name_and_historical_nickname(self) -> None:
        for team_input, expected in (
            ("TB", "TB"),
            ("Tampa Bay Buccaneers", "TB"),
            ("Raiders", "LV"),
            ("OAK", "LV"),
        ):
            with self.subTest(team=team_input):
                result = self.service.latest(NewsQuery(team=team_input))
                self.assertEqual(result.outcome, NewsOutcome.SUCCESS)
                self.assertEqual(result.resolved_team.code, expected)

    def test_ambiguous_city_returns_candidates(self) -> None:
        result = self.service.latest(NewsQuery(team="Los Angeles"))

        self.assertEqual(result.outcome, NewsOutcome.TEAM_AMBIGUOUS)
        self.assertEqual(
            {team.code for team in result.team_candidates}, {"LA", "LAC"}
        )
        self.assertEqual(self.repository.report_calls, [])

    def test_team_typo_returns_retry_suggestion_without_guessing(self) -> None:
        result = self.service.latest(NewsQuery(team="Vikngs"))

        self.assertEqual(result.outcome, NewsOutcome.TEAM_NOT_FOUND)
        self.assertEqual(result.team_candidates[0].code, "MIN")
        self.assertEqual(self.repository.report_calls, [])

    def test_full_view_is_capped(self) -> None:
        self.repository.reports = [make_report(str(index)) for index in range(8)]

        result = self.service.latest(
            NewsQuery(count=8, detail=NewsDetail.FULL)
        )

        self.assertEqual(result.outcome, NewsOutcome.SUCCESS)
        self.assertTrue(result.full_view_capped)
        self.assertEqual(len(result.reports), MAX_FULL_NEWS_COUNT)
        self.assertEqual(
            self.repository.report_calls[0]["count"], MAX_FULL_NEWS_COUNT
        )

    def test_no_reports_is_an_explicit_outcome(self) -> None:
        self.repository.reports = []

        result = self.service.latest(NewsQuery(team="ATL"))

        self.assertEqual(result.outcome, NewsOutcome.NO_REPORTS)
        self.assertEqual(result.resolved_team.code, "ATL")

    def test_query_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 10"):
            NewsQuery(count=11)
        with self.assertRaisesRegex(ValueError, "detail"):
            NewsQuery(detail="summary")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "blank"):
            NewsQuery(player=" ")


if __name__ == "__main__":
    unittest.main()
