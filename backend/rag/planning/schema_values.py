"""Finite database vocabularies exposed to the RAG query planner.

These values mirror the processed datasets loaded into Supabase. Keep this
catalog small and stable: player names, colleges, storylines, and other open
text values intentionally do not belong here.
"""

from enum import StrEnum


class TeamCode(StrEnum):
    """Canonical nflverse team codes plus genuine historical franchises."""

    ARI = "ARI"
    ATL = "ATL"
    BAL = "BAL"
    BUF = "BUF"
    CAR = "CAR"
    CHI = "CHI"
    CIN = "CIN"
    CLE = "CLE"
    DAL = "DAL"
    DEN = "DEN"
    DET = "DET"
    GB = "GB"
    HOU = "HOU"
    IND = "IND"
    JAX = "JAX"
    KC = "KC"
    LA = "LA"
    LAC = "LAC"
    LV = "LV"
    MIA = "MIA"
    MIN = "MIN"
    NE = "NE"
    NO = "NO"
    NYG = "NYG"
    NYJ = "NYJ"
    PHI = "PHI"
    PIT = "PIT"
    SEA = "SEA"
    SF = "SF"
    TB = "TB"
    TEN = "TEN"
    WAS = "WAS"

    # Preserve real historical identities rather than rewriting old seasons.
    OAK = "OAK"
    SD = "SD"
    STL = "STL"


class PlayerPosition(StrEnum):
    C = "C"
    CB = "CB"
    DB = "DB"
    DE = "DE"
    DL = "DL"
    DT = "DT"
    FB = "FB"
    FS = "FS"
    G = "G"
    ILB = "ILB"
    K = "K"
    LB = "LB"
    LS = "LS"
    MLB = "MLB"
    NT = "NT"
    OL = "OL"
    OLB = "OLB"
    OT = "OT"
    P = "P"
    QB = "QB"
    RB = "RB"
    S = "S"
    SAF = "SAF"
    TE = "TE"
    WR = "WR"


class PositionGroup(StrEnum):
    DB = "DB"
    DL = "DL"
    LB = "LB"
    OL = "OL"
    QB = "QB"
    RB = "RB"
    SPEC = "SPEC"
    TE = "TE"
    WR = "WR"


class RosterStatus(StrEnum):
    ACT = "ACT"
    CUT = "CUT"
    DEV = "DEV"
    EXE = "EXE"
    INA = "INA"
    NWT = "NWT"
    PUP = "PUP"
    RES = "RES"
    RET = "RET"
    RLS = "RLS"
    RSN = "RSN"
    RSR = "RSR"
    SUS = "SUS"


class DepthChartPosition(StrEnum):
    C = "C"
    FB = "FB"
    FS = "FS"
    H = "H"
    KR = "KR"
    LCB = "LCB"
    LDE = "LDE"
    LDT = "LDT"
    LG = "LG"
    LILB = "LILB"
    LS = "LS"
    LT = "LT"
    MLB = "MLB"
    NB = "NB"
    NT = "NT"
    P = "P"
    PK = "PK"
    PR = "PR"
    QB = "QB"
    RB = "RB"
    RCB = "RCB"
    RDE = "RDE"
    RDT = "RDT"
    RG = "RG"
    RILB = "RILB"
    RT = "RT"
    SLB = "SLB"
    SS = "SS"
    TE = "TE"
    WLB = "WLB"
    WR = "WR"


class Formation(StrEnum):
    DEFENSE = "Defense"
    OFFENSE = "Offense"
    SPECIAL_TEAMS = "Special Teams"


class ECRPosition(StrEnum):
    DB = "DB"
    DL = "DL"
    K = "K"
    LB = "LB"
    QB = "QB"
    RB = "RB"
    TE = "TE"
    WR = "WR"


class ECRScoringFormat(StrEnum):
    PPR = "ppr"
    SOURCE_DEFAULT = "source_default"


class ECRLeagueFormat(StrEnum):
    BEST_BALL = "best_ball"
    DYNASTY_1QB = "dynasty_1qb"
    DYNASTY_IDP = "dynasty_idp"
    DYNASTY_ROOKIE = "dynasty_rookie"
    DYNASTY_SUPERFLEX = "dynasty_superflex"
    REDRAFT_1QB = "redraft_1qb"
    REDRAFT_IDP = "redraft_idp"
    REDRAFT_SUPERFLEX = "redraft_superflex"


class Conference(StrEnum):
    AFC = "AFC"
    NFC = "NFC"


class Division(StrEnum):
    AFC_EAST = "AFC East"
    AFC_NORTH = "AFC North"
    AFC_SOUTH = "AFC South"
    AFC_WEST = "AFC West"
    NFC_EAST = "NFC East"
    NFC_NORTH = "NFC North"
    NFC_SOUTH = "NFC South"
    NFC_WEST = "NFC West"
