"""2027 projection: one row per (batter, season) of contact-stage inputs (S2).

A season-level table is Contact Forecast's window features computed over a
hitter's WHOLE season instead of his first K batted balls. It reuses
`forecast.features._hitter_season_features` unchanged, so every feature has
the same definition Model D was built on. Column names keep Contact
Forecast's `std_*` / `*_through_cutoff` spelling so `forecast.ridge` can
consume them directly; here "through cutoff" means "through the season".

Only resolved batted balls count (`forecast.windows.select_scored_eligible_
events`, production's per-100 denominator). This module applies no season
guard of its own: the caller decides which seasons it may read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecast.features import _hitter_season_features
from forecast.forecast_config import (
    CHRONOLOGICAL_SORT_KEY,
    DESERVED_COLUMN,
    OBSERVED_COLUMN,
    SURPRISE_COLUMN,
)
from forecast.windows import select_scored_eligible_events

#: Same tolerance as `forecast.windows._IDENTITY_ATOL`.
_IDENTITY_ATOL = 1e-6

_RECENT_FEATURES = (
    "recent_realized_rv_per_100",
    "recent_deserved_rv_per_100",
    "recent_surprise_rv_per_100",
)


class ProjectionInputError(ValueError):
    """Raised when a ledger cannot support a season-level input table."""


def resolved_events(ledger: pd.DataFrame) -> pd.DataFrame:
    """Resolved batted balls for exactly one season, identity-checked."""
    seasons = sorted(int(s) for s in ledger["season"].unique())
    if len(seasons) != 1:
        raise ProjectionInputError(f"Expected exactly one season; got {seasons}")
    kept, _ = select_scored_eligible_events(ledger)
    residual = (
        kept[OBSERVED_COLUMN].to_numpy(dtype=float)
        - kept[DESERVED_COLUMN].to_numpy(dtype=float)
        - kept[SURPRISE_COLUMN].to_numpy(dtype=float)
    )
    worst = float(np.max(np.abs(residual))) if residual.size else 0.0
    if worst > _IDENTITY_ATOL:
        raise ProjectionInputError(
            "Contact-stage identity violated (Rc - E0 != surprise, max residual "
            f"{worst:.3e}); this is not the frozen Contact Luck methodology."
        )
    return kept


def build_season_inputs(ledger: pd.DataFrame) -> pd.DataFrame:
    """One row per batter: resolved-BBE count, realized/deserved RV/100, contact profile."""
    events = resolved_events(ledger)
    season = int(events["season"].iloc[0])
    ordered = events.sort_values(["batter", *CHRONOLOGICAL_SORT_KEY], kind="mergesort")

    rows: list[dict[str, Any]] = []
    for batter, group in ordered.groupby("batter", sort=True):
        record: dict[str, Any] = {"batter": batter, "season": season}
        features = _hitter_season_features(group)
        for name in _RECENT_FEATURES:
            features.pop(name, None)
        record.update(features)
        rows.append(record)
    return pd.DataFrame(rows)
