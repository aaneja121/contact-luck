"""2027 projection: the one challenger (S5).

Contact Forecast's Model D ridge, refit at season level with
`CHALLENGER_NOMINAL_FEATURES` (Model D minus its within-season recent-window
features). Everything that makes `forecast.ridge` safe is reused unchanged:
the prespecified drops, the identifiability check, training-only
standardization, and `assert_no_later_season_rows`.

## Penalty selection

Chosen by batter-grouped `CV_FOLDS`-fold cross-validation inside the
challenger's fit pair (2022->2023) only, over `forecast.ridge.ALPHA_GRID`, on
MAE. `GroupKFold` without shuffling is deterministic. The model is then
refit on every complete-case row of that pair at the selected penalty.

## Incomplete features

A hitter missing any retained feature is excluded from training (counted).
At prediction time he receives the primary benchmark's value as his
challenger projection (counted), so the paired comparison covers every
graded hitter and the challenger can neither gain nor lose on rows it
cannot score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from forecast.ridge import (
    ALPHA_GRID,
    SELECTION_METRIC,
    RidgeFit,
    assert_no_later_season_rows,
    fit_ridge,
    predict,
)
from projection.projection_config import (
    CHALLENGER_COLUMN,
    CHALLENGER_NOMINAL_FEATURES,
    CV_FOLDS,
    TARGET_COLUMN,
)


def complete_cases(frame: pd.DataFrame) -> pd.Series:
    """True where every nominal challenger feature is finite."""
    values = frame[list(CHALLENGER_NOMINAL_FEATURES)].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(values.to_numpy(dtype=float)).all(axis=1), index=frame.index)


def _fit(train: pd.DataFrame, alpha: float) -> RidgeFit:
    return fit_ridge(
        train,
        model_name=CHALLENGER_COLUMN,
        alpha=alpha,
        target=TARGET_COLUMN,
        nominal_features=CHALLENGER_NOMINAL_FEATURES,
    )


def select_alpha_grouped_cv(train: pd.DataFrame) -> dict[str, Any]:
    """Batter-grouped CV over the frozen alpha grid; returns the full record."""
    rows = train.loc[complete_cases(train)].reset_index(drop=True)
    splitter = GroupKFold(n_splits=CV_FOLDS)
    splits = list(splitter.split(rows, groups=rows["batter"]))
    y = rows[TARGET_COLUMN].to_numpy(dtype=float)

    grid: list[dict[str, float]] = []
    for alpha in ALPHA_GRID:
        abs_errors = np.empty(len(rows))
        for train_idx, validate_idx in splits:
            fit = _fit(rows.iloc[train_idx], alpha)
            abs_errors[validate_idx] = np.abs(
                predict(fit, rows.iloc[validate_idx]) - y[validate_idx]
            )
        grid.append({"alpha": float(alpha), "mae": float(abs_errors.mean())})

    best = min(grid, key=lambda row: row[SELECTION_METRIC])
    return {
        "selection_metric": SELECTION_METRIC,
        "n_folds": CV_FOLDS,
        "alpha_grid": [float(a) for a in ALPHA_GRID],
        "selected_alpha": best["alpha"],
        "selected_alpha_at_grid_boundary": best["alpha"] in (ALPHA_GRID[0], ALPHA_GRID[-1]),
        "grid": grid,
        "n_rows": int(len(rows)),
        "n_excluded_incomplete_features": int(len(train) - len(rows)),
        "train_seasons": sorted(int(s) for s in rows["season"].unique()),
        "folds": [
            {
                "train_batters": sorted(rows["batter"].iloc[t].tolist()),
                "validation_batters": sorted(rows["batter"].iloc[v].tolist()),
            }
            for t, v in splits
        ],
    }


def fit_challenger(train: pd.DataFrame, *, alpha: float, evaluation_input_season: int) -> RidgeFit:
    """Fit on complete cases, refusing any row from the evaluation's input season or later."""
    assert_no_later_season_rows(train, evaluation_input_season)
    return _fit(train.loc[complete_cases(train)], alpha)


def predict_with_fallback(
    fit: RidgeFit, frame: pd.DataFrame, *, fallback_column: str
) -> tuple[pd.Series, int]:
    """Challenger predictions; incomplete-feature rows take `fallback_column`'s value."""
    complete = complete_cases(frame)
    values = frame[fallback_column].astype(float).copy()
    if bool(complete.any()):
        values.loc[complete] = predict(fit, frame.loc[complete])
    return values, int((~complete).sum())
