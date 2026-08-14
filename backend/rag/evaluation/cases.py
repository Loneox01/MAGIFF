from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCase:
    name: str
    query: str
    expected_ids: frozenset[str]


CASES = [
    RetrievalCase(
        name="Sadiq setback",
        query="What is the latest update on Kenyon Sadiq's hernia setback?",
        expected_ids=frozenset(
            {"nbc-sports-2026-08-04-kenyon-sadiq-recovery-setback"}
        ),
    ),
    RetrievalCase(
        name="Seattle backfield injury",
        query="Jadarian Price lower body soreness Seattle backfield",
        expected_ids=frozenset(
            {"the-news-tribune-2026-08-08-jadarian-price-lower-body-soreness"}
        ),
    ),
    RetrievalCase(
        name="McConkey availability",
        query="Was Ladd McConkey full go for Chargers training camp?",
        expected_ids=frozenset(
            {"los-angeles-chargers-2026-07-28-ladd-mcconkey-full-go"}
        ),
    ),
    RetrievalCase(
        name="Lemon missed practice",
        query="Makai Lemon missed practice with a hamstring injury",
        expected_ids=frozenset(
            {"nbc-sports-2026-08-08-makai-lemon-misses-practice"}
        ),
    ),
    RetrievalCase(
        name="Latest Price recovery step",
        query="What is the latest Jadarian Price injury update?",
        expected_ids=frozenset(
            {"fantasypros-2026-08-12-jadarian-price-walkthrough"}
        ),
    ),
    RetrievalCase(
        name="Mahomes Week 1 health",
        query="Is Patrick Mahomes healthy for Week 1?",
        expected_ids=frozenset(
            {
                "nfl-2026-07-24-patrick-mahomes-cleared",
                "fantasypros-2026-08-13-patrick-mahomes-sits-preseason",
            }
        ),
    ),
    RetrievalCase(
        name="Browns quarterback competition",
        query="Who is winning the Browns quarterback competition?",
        expected_ids=frozenset(
            {
                "fantasypros-2026-08-12-deshaun-watson-preseason-start",
                "fantasypros-2026-08-12-shedeur-sanders-preseason-role",
            }
        ),
    ),
    RetrievalCase(
        name="Bucky Irving full clearance",
        query="Is Bucky Irving fully healthy?",
        expected_ids=frozenset(
            {"tampa-bay-buccaneers-2026-07-29-bucky-irving-full-go"}
        ),
    ),
    RetrievalCase(
        name="Chargers lead running back",
        query="Which Chargers running back is expected to lead the backfield?",
        expected_ids=frozenset({"nfl-2026-06-17-omarion-hampton-lead-back"}),
    ),
]
