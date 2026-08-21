"""Typed structured-data requests used to enrich report retrieval.

The report planner emits these requests once. Application code resolves their
anchors, invokes the existing read-only NFL tools, and reduces the results into
bounded retrieval context. They are deliberately higher level than public tool
calls: the planner refers to an already-declared selector instead of inventing
internal player IDs or repeating team membership from model memory.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tools.field_catalog import (
    PLAYER_FORMULA_FIELDS,
    PLAYER_SEASON_STAT_FIELDS,
    PLAYER_WEEKLY_STAT_FIELDS,
    TEAM_WEEKLY_STAT_FIELDS,
    field_names,
)
from tools.formulas import parse_formula
from tools.team_analytics import (
    TEAM_DEFENSE_FORMULA_FIELDS,
    TEAM_OFFENSE_FORMULA_FIELDS,
)

from .schema_values import (
    ECRLeagueFormat,
    ECRPosition,
    ECRScoringFormat,
    PlayerPosition,
    RosterStatus,
)


class LookupModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LookupPurpose(StrEnum):
    """Why structured information is needed by report retrieval."""

    RESOLVE_RELATIONSHIP = "resolve_relationship"
    EXPAND_CANDIDATES = "expand_candidates"
    ENRICH_QUERY = "enrich_query"
    RERANKER_CONTEXT = "reranker_context"


class ContextScopePolicy(StrEnum):
    """Branch-local metadata boundary for contextual report retrieval."""

    ANCHOR_TEAMS = "anchor_teams"
    ANCHOR_AND_LOOKUP_TEAMS = "anchor_and_lookup_teams"
    LOOKUP_ENTITIES = "lookup_entities"
    SEMANTIC_ONLY = "semantic_only"


TEAM_ANCHORED_LOOKUP_OPERATIONS = frozenset(
    {
        "team_roster",
        "team_depth_chart",
        "team_schedule",
        "team_weekly_stats",
    }
)
PLAYER_ANCHORED_LOOKUP_OPERATIONS = frozenset(
    {
        "player_season_stats",
        "player_weekly_stats",
        "player_snap_counts",
    }
)
PLAYER_RANKING_LOOKUP_OPERATIONS = frozenset(
    {"ecr_ranking", "player_formula_ranking"}
)
TEAM_RANKING_LOOKUP_OPERATIONS = frozenset({"team_formula_ranking"})


PlayerSeasonField = Literal[*tuple(sorted(PLAYER_SEASON_STAT_FIELDS))]
PlayerWeeklyField = Literal[*tuple(sorted(PLAYER_WEEKLY_STAT_FIELDS))]
TeamWeeklyField = Literal[*tuple(sorted(TEAM_WEEKLY_STAT_FIELDS))]
PlayerFormulaField = Literal[*tuple(sorted(PLAYER_FORMULA_FIELDS))]


class LookupBase(LookupModel):
    lookup_id: str = Field(
        min_length=1,
        max_length=64,
        description="Unique stable identifier within this query plan.",
    )
    purpose: LookupPurpose = Field(
        description=(
            "Reason this lookup is necessary. This records provenance; it does "
            "not by itself authorize a global report filter."
        )
    )


class TeamRosterLookup(LookupBase):
    operation: Literal["team_roster"]
    season: int = Field(ge=1920, le=2100)
    week: int | None = Field(ge=1, le=22)
    position: PlayerPosition | None
    status: RosterStatus | None


class TeamDepthChartLookup(LookupBase):
    operation: Literal["team_depth_chart"]
    season: int = Field(ge=1920, le=2100)
    week: int | None = Field(ge=1, le=22)
    position: PlayerPosition | None


class TeamScheduleLookup(LookupBase):
    operation: Literal["team_schedule"]
    season: int = Field(ge=1920, le=2100)
    week: int | None = Field(ge=1, le=22)


class PlayerSeasonStatsLookup(LookupBase):
    operation: Literal["player_season_stats"]
    season: int = Field(ge=1920, le=2100)
    season_type: Literal["REG", "POST"]
    fields: list[PlayerSeasonField] | None = Field(
        max_length=12,
        description=(
            "Stored season fields needed to interpret report retrieval. Null "
            "uses the structured tool's compact defaults."
        ),
    )


class PlayerWeeklyStatsLookup(LookupBase):
    operation: Literal["player_weekly_stats"]
    season: int = Field(ge=1920, le=2100)
    week: int | None = Field(ge=1, le=22)
    fields: list[PlayerWeeklyField] | None = Field(
        max_length=12,
        description=(
            "Stored weekly fields needed for the retrieval context. Null uses "
            "the structured tool's compact defaults."
        ),
    )


class PlayerSnapCountsLookup(LookupBase):
    operation: Literal["player_snap_counts"]
    season: int = Field(ge=1920, le=2100)
    week: int | None = Field(ge=1, le=22)


class TeamWeeklyStatsLookup(LookupBase):
    operation: Literal["team_weekly_stats"]
    season: int = Field(ge=1920, le=2100)
    week: int | None = Field(ge=1, le=22)
    fields: list[TeamWeeklyField] | None = Field(
        max_length=12,
        description=(
            "Stored team fields needed for retrieval context. Null uses the "
            "structured tool's compact defaults."
        ),
    )


class ECRRankingLookup(LookupBase):
    operation: Literal["ecr_ranking"]
    season: int = Field(ge=1920, le=2100)
    positions: list[ECRPosition] | None = Field(min_length=1, max_length=4)
    scoring_format: ECRScoringFormat
    league_format: ECRLeagueFormat
    snapshot_type: Literal[
        "current",
        "final_preseason",
        "season_opening",
    ]
    as_of_date: str | None
    minimum_overall_rank: float | None = Field(ge=1)
    maximum_overall_rank: float | None = Field(ge=1)
    limit: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _validate_rank_range(self):
        if (
            self.minimum_overall_rank is not None
            and self.maximum_overall_rank is not None
            and self.minimum_overall_rank > self.maximum_overall_rank
        ):
            raise ValueError(
                "minimum_overall_rank cannot exceed maximum_overall_rank"
            )
        return self


class PlayerFormulaRankingLookup(LookupBase):
    operation: Literal["player_formula_ranking"]
    season: int = Field(ge=1920, le=2100)
    formula: str = Field(
        min_length=1,
        max_length=160,
        description=(
            "Arithmetic expression over stored player-season fields only: "
            f"{field_names(PLAYER_FORMULA_FIELDS)}."
        ),
    )
    season_type: Literal["REG", "POST"]
    position: PlayerPosition | None
    minimum_field: PlayerFormulaField | None
    minimum_value: float | None = Field(ge=0)
    sort_direction: Literal["asc", "desc"]
    limit: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _validate_formula_vocabulary(self):
        if (self.minimum_field is None) != (self.minimum_value is None):
            raise ValueError(
                "minimum_field and minimum_value must both be set or null"
            )
        parse_formula(self.formula, set(PLAYER_FORMULA_FIELDS))
        return self


class TeamFormulaRankingLookup(LookupBase):
    operation: Literal["team_formula_ranking"]
    season: int = Field(ge=1920, le=2100)
    perspective: Literal["offense", "defense"]
    formula: str = Field(
        min_length=1,
        max_length=160,
        description=(
            "Arithmetic expression over fields allowed for the selected "
            "perspective. Offense: "
            f"{field_names(TEAM_OFFENSE_FORMULA_FIELDS)}. Defense: "
            f"{field_names(TEAM_DEFENSE_FORMULA_FIELDS)}."
        ),
    )
    season_type: Literal["REG", "POST"]
    minimum_games: int | None = Field(ge=1)
    sort_direction: Literal["asc", "desc"]
    limit: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _validate_formula_vocabulary(self):
        allowed = (
            TEAM_OFFENSE_FORMULA_FIELDS
            if self.perspective == "offense"
            else TEAM_DEFENSE_FORMULA_FIELDS
        )
        parse_formula(self.formula, allowed)
        return self


StructuredLookup = (
    TeamRosterLookup
    | TeamDepthChartLookup
    | TeamScheduleLookup
    | PlayerSeasonStatsLookup
    | PlayerWeeklyStatsLookup
    | PlayerSnapCountsLookup
    | TeamWeeklyStatsLookup
    | ECRRankingLookup
    | PlayerFormulaRankingLookup
    | TeamFormulaRankingLookup
)


__all__ = [
    "ContextScopePolicy",
    "LookupPurpose",
    "PLAYER_ANCHORED_LOOKUP_OPERATIONS",
    "PLAYER_RANKING_LOOKUP_OPERATIONS",
    "StructuredLookup",
    "TEAM_ANCHORED_LOOKUP_OPERATIONS",
    "TEAM_RANKING_LOOKUP_OPERATIONS",
]
