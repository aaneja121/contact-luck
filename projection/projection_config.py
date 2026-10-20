"""2027 projection: season roles, frozen constants, namespaces and season guards.

The specification is `docs/plans/2027_projection_model_plan.md` (S1-S8, frozen
2026-10-08). This module holds its numbers; it does not restate its reasoning.

## Season roles

| Season | Role here |
|---|---|
| 2022->2023, 2023->2024 | The two development pairs. The only seasons this stage reads. |
| 2025 | SEALED. Never read, under any flag. |
| 2026 | Projection INPUT. Scored once by `projection.run_2026_scoring`; unreachable from development code. |
| 2027 | Prospective TEST. Read once, after the season, by the A3 grader only (not yet written). |

## Why this lives outside `src/` and `prospective/`

The v1.3 2027 freeze (`d5303bc`) forbids changing either after that commit.
`projection/` only imports from `forecast/` and changes nothing it imports.
"""

from __future__ import annotations

from pathlib import Path

from forecast.forecast_config import FORBIDDEN_NAMESPACE_ROOTS, PROJECT_ROOT
from forecast.ridge import FEATURE_SETS

# --------------------------------------------------------------------------
# Season roles
# --------------------------------------------------------------------------

#: (input season, target season). Deserved values are walk-forward: 2022 was
#: scored by a contact model fit on 2021 only, 2023 on 2021-22.
DEVELOPMENT_PAIRS: tuple[tuple[int, int], ...] = ((2022, 2023), (2023, 2024))

#: S5: the challenger is fit, and its penalty chosen, on this pair only.
CHALLENGER_FIT_PAIR: tuple[int, int] = (2022, 2023)

#: S6: the adoption rule is evaluated once, on this pair.
ADOPTION_EVALUATION_PAIR: tuple[int, int] = (2023, 2024)

DEVELOPMENT_SEASONS: tuple[int, ...] = (2022, 2023, 2024)
SEALED_SEASONS: tuple[int, ...] = (2025,)
PROJECTION_INPUT_SEASON = 2026
PROJECTION_TEST_SEASON = 2027

# --------------------------------------------------------------------------
# Frozen constants (S1-S7 and the implementation clarifications)
# --------------------------------------------------------------------------

#: S1 / decision 4: a projected hitter is graded at >= this many resolved
#: batted balls in the target season.
GRADING_MIN_RESOLVED_BBE = 100

#: S4: hitter-seasons below this contribute to the within-hitter variance but
#: not to the between-hitter estimate. `forecast.baselines.fit_shrinkage`'s
#: own default, carried over unchanged.
SHRINKAGE_MIN_EVENTS_PER_HITTER = 50

#: S5: batter-grouped cross-validation folds for the ridge penalty.
CV_FOLDS = 5

#: S6 and S7: paired, batter-clustered percentile bootstrap. S7 fixes 500
#: replicates and seed 42 (the v1.3 pre-registration's convention); S6 uses
#: the same.
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 42
BOOTSTRAP_ALPHA = 0.05

#: S3: the ladder, in report order. Names match Contact Forecast's rungs.
LADDER: tuple[str, ...] = (
    "league_mean",
    "shrunk_realized_persistence",
    "shrunk_deserved_persistence",
)
PRIMARY_BENCHMARK = "shrunk_deserved_persistence"
REALIZED_BENCHMARK = "shrunk_realized_persistence"
CHALLENGER_COLUMN = "challenger_d_season"

#: S5: Contact Forecast's Model D, minus its within-season "recent window"
#: features, which have no season-level equivalent. `forecast.ridge`'s
#: PRESPECIFIED_DROPS (the surprise column, the popup rate) still apply.
CHALLENGER_NOMINAL_FEATURES: tuple[str, ...] = tuple(
    f for f in FEATURE_SETS["D_full_contact_profile"] if not f.startswith("recent_")
)

#: S2: the target, per 100 resolved batted balls in the target season.
TARGET_COLUMN = "target_realized_rv_per_100"

#: Required disclosure (S6).
DESIGN_TIME_DISCLOSURE = (
    "The 2026 Contact Forecast result was known at design time: the resolution pass "
    "(2026-09-29) found no evidence that Model D improves on shrunk deserved. The ladder "
    "and the adoption rule were fixed after that result was seen."
)

# --------------------------------------------------------------------------
# Namespaces
# --------------------------------------------------------------------------

DEVELOPMENT_LEDGER_DIR = PROJECT_ROOT / "data" / "forecast"
PROJECTION_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "projection_2027"
DEVELOPMENT_OUTPUT_DIR = PROJECTION_OUTPUTS_DIR / "development"
SPEC_PATH = PROJECT_ROOT / "docs" / "plans" / "2027_projection_model_plan.md"
DEVELOPMENT_RESULTS_PATH = DEVELOPMENT_OUTPUT_DIR / "development_results.json"
PROJECTIONS_OUTPUT_DIR = PROJECTION_OUTPUTS_DIR / "projections"

# --------------------------------------------------------------------------
# The 2026 input (S2, S8)
# --------------------------------------------------------------------------

#: S2: the archived v1.1 snapshot through the last 2026 regular-season day,
#: pinned by label and verified against its own integrity hashes.
PROJECTION_INPUT_SNAPSHOT_LABEL = "2026-09-27"
PROJECTION_INPUT_DATA_THROUGH = "2026-09-27"


def development_ledger_path(season: int) -> Path:
    """The walk-forward ledger `forecast.score_development_seasons` wrote."""
    assert_projection_development_seasons([season])
    return DEVELOPMENT_LEDGER_DIR / f"walk_forward_ledger_{int(season)}.parquet"


class ProjectionSeasonError(ValueError):
    """Raised when development code is asked for a season it may not read."""


class ProjectionNamespaceError(ValueError):
    """Raised when a path resolves inside a sealed or prospective namespace."""


def assert_projection_development_seasons(seasons: list[int] | tuple[int, ...]) -> None:
    """Refuse every season outside the two development pairs.

    No override argument exists. The 2026 input and the 2027 grade each get
    their own separately reviewed entry point.
    """
    years = {int(s) for s in seasons}
    if years & set(SEALED_SEASONS):
        raise ProjectionSeasonError(
            f"{sorted(years & set(SEALED_SEASONS))} is sealed final-evaluation data; the "
            "projection model never reads it (decision 1)."
        )
    if PROJECTION_INPUT_SEASON in years:
        raise ProjectionSeasonError(
            f"{PROJECTION_INPUT_SEASON} is the projection input season. It is scored once, "
            "by its own entry point, after development is frozen -- never by development code."
        )
    if PROJECTION_TEST_SEASON in years:
        raise ProjectionSeasonError(
            f"{PROJECTION_TEST_SEASON} is the prospective test season. Only the A3 grader "
            "may read it, once, after the season ends (RESEARCH_RULES.md)."
        )
    unknown = years - set(DEVELOPMENT_SEASONS)
    if unknown:
        raise ProjectionSeasonError(
            f"{sorted(unknown)} is not a projection development season "
            f"(authorized: {list(DEVELOPMENT_SEASONS)})."
        )


def assert_path_outside_forbidden_namespaces(path: Path) -> None:
    """Refuse a path inside the sealed 2025 or the prospective 2026/2027 roots."""
    resolved = Path(path).resolve()
    for root in FORBIDDEN_NAMESPACE_ROOTS:
        if resolved == root or root in resolved.parents:
            raise ProjectionNamespaceError(
                f"{resolved} resolves inside the forbidden namespace {root}; projection "
                "development reads development ledgers only."
            )


def assert_projection_input_season(seasons: list[int] | tuple[int, ...]) -> None:
    """Allow the projection input season, 2026, and nothing else.

    Used only by the 2026 scoring entry point. Development code keeps
    `assert_projection_development_seasons`, which refuses 2026.
    """
    years = {int(s) for s in seasons}
    if years != {PROJECTION_INPUT_SEASON}:
        raise ProjectionSeasonError(
            f"The 2027 projection is scored from {PROJECTION_INPUT_SEASON} inputs only; "
            f"got season(s) {sorted(years)}."
        )
