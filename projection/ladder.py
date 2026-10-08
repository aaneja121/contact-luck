"""2027 projection: the ladder rungs (S3) with input-season shrinkage (S4, option a).

One estimator for realized and deserved: `forecast.baselines.fit_shrinkage`,
unchanged. Its parameters are estimated on the INPUT season's resolved
batted balls only -- 2022 for 2022->2023, 2023 for 2023->2024, 2026 for the
2027 projection (RESEARCH_RULES.md, "Shrinkage parameters may be estimated on
2026 inputs"). No target-season row is ever seen, and
`forecast.baselines.assert_fit_is_causal` re-proves that on every fit.

At season level `n_i` differs by hitter, so shrinkage CAN change rank; that
is why Spearman is reported (S4).
"""

from __future__ import annotations

import pandas as pd

from forecast.baselines import ShrinkageFit, assert_fit_is_causal, fit_shrinkage
from forecast.forecast_config import DESERVED_COLUMN, OBSERVED_COLUMN
from projection.projection_config import SHRINKAGE_MIN_EVENTS_PER_HITTER
from projection.season_table import ProjectionInputError, resolved_events


def fit_input_season_shrinkage(
    input_ledger: pd.DataFrame, *, target_season: int
) -> tuple[ShrinkageFit, ShrinkageFit]:
    """Fit realized and deserved shrinkage on the input season, the season before the target."""
    seasons = sorted(int(s) for s in input_ledger["season"].unique())
    if seasons != [int(target_season) - 1]:
        raise ProjectionInputError(
            f"Shrinkage must be fit on the input season {int(target_season) - 1} only "
            f"(S4 option a); got season(s) {seasons}"
        )
    events = resolved_events(input_ledger)
    realized = fit_shrinkage(
        events,
        value_column=OBSERVED_COLUMN,
        quantity="realized",
        min_events_per_hitter=SHRINKAGE_MIN_EVENTS_PER_HITTER,
    )
    deserved = fit_shrinkage(
        events,
        value_column=DESERVED_COLUMN,
        quantity="deserved",
        min_events_per_hitter=SHRINKAGE_MIN_EVENTS_PER_HITTER,
    )
    assert_fit_is_causal(realized, target_season)
    assert_fit_is_causal(deserved, target_season)
    return realized, deserved


def add_ladder(
    inputs: pd.DataFrame, *, realized_fit: ShrinkageFit, deserved_fit: ShrinkageFit
) -> pd.DataFrame:
    """Add every ladder rung to a season-level input (or pair) table."""
    out = inputs.copy()
    n_events = out["n_resolved_bbe_through_cutoff"]
    out["league_mean"] = realized_fit.prior_mean
    out["shrunk_realized_persistence"] = realized_fit.apply(
        out["std_realized_rv_per_100"], n_events
    )
    out["shrunk_deserved_persistence"] = deserved_fit.apply(
        out["std_deserved_rv_per_100"], n_events
    )
    return out
