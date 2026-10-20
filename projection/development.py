"""2027 projection: the development stage (S1, S3-S6), run once on 2022-2024.

`evaluate_development` is a pure function of the three walk-forward ledgers.
It builds both development pairs, reports the ladder on each as descriptive
results, fits the challenger on 2022->2023, and applies the S6 adoption rule
once, on 2023->2024:

    adopt the challenger only if delta_MAE (challenger - shrunk deserved) < 0
    AND its batter-clustered bootstrap 95% CI lies entirely below zero.

Otherwise shrunk deserved ships. The bootstrap is `forecast.bootstrap.
run_paired_bootstrap`, unchanged, on one row per graded hitter; its
(cutoff, horizon) grid is collapsed to the single label (0, 0), meaning
"full input season -> full target season".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from forecast.baselines import shrinkage_fit_record
from forecast.bootstrap import BootstrapDesign, run_paired_bootstrap
from forecast.metrics import evaluate_ladder
from projection.challenger import (
    fit_challenger,
    predict_with_fallback,
    select_alpha_grouped_cv,
)
from projection.ladder import add_ladder, fit_input_season_shrinkage
from projection.projection_config import (
    ADOPTION_EVALUATION_PAIR,
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CHALLENGER_COLUMN,
    CHALLENGER_FIT_PAIR,
    DESIGN_TIME_DISCLOSURE,
    DEVELOPMENT_PAIRS,
    GRADING_MIN_RESOLVED_BBE,
    LADDER,
    PRIMARY_BENCHMARK,
    TARGET_COLUMN,
    assert_projection_development_seasons,
)
from projection.season_table import build_season_inputs

ADOPTION_RULE = (
    "adopt the challenger only if delta_MAE (challenger - shrunk deserved) < 0 on "
    "2023->2024 AND its batter-clustered bootstrap 95% CI lies entirely below zero; "
    "otherwise shrunk deserved ships"
)

#: The single bootstrap cell label: full input season -> full target season.
_CELL = "K0_H0"


@dataclass(frozen=True)
class PairTables:
    """One development pair: everyone projected, the graded subset, and the counts."""

    projected: pd.DataFrame
    graded: pd.DataFrame
    survivorship: dict[str, int]


def build_pair(inputs: pd.DataFrame, targets: pd.DataFrame) -> PairTables:
    """S1: project every input-season hitter; grade those with >= 100 target BBE."""
    target = targets[["batter", "n_resolved_bbe_through_cutoff", "std_realized_rv_per_100"]].rename(
        columns={
            "n_resolved_bbe_through_cutoff": "target_n_resolved_bbe",
            "std_realized_rv_per_100": TARGET_COLUMN,
        }
    )
    projected = inputs.merge(target, on="batter", how="left", validate="one_to_one")
    qualified = projected["target_n_resolved_bbe"] >= GRADING_MIN_RESOLVED_BBE
    graded = projected.loc[qualified].reset_index(drop=True)

    target_qualified = set(
        target.loc[target["target_n_resolved_bbe"] >= GRADING_MIN_RESOLVED_BBE, "batter"]
    )
    return PairTables(
        projected=projected,
        graded=graded,
        survivorship={
            "n_projected": int(len(projected)),
            "n_projected_with_any_target_bbe": int(
                projected["target_n_resolved_bbe"].notna().sum()
            ),
            "n_graded": int(len(graded)),
            "n_projected_not_graded": int(len(projected) - len(graded)),
            "n_target_qualified_without_input_row": len(target_qualified - set(inputs["batter"])),
        },
    )


def adoption_decision(delta_mae: dict[str, Any]) -> dict[str, Any]:
    """S6, applied to the pooled paired delta record from the bootstrap."""
    point = float(delta_mae["point_estimate"])
    upper = delta_mae["ci_upper"]
    adopt = bool(point < 0.0 and upper is not None and float(upper) < 0.0)
    return {
        "rule": ADOPTION_RULE,
        "delta_mae_point": point,
        "delta_mae_ci_lower": delta_mae["ci_lower"],
        "delta_mae_ci_upper": upper,
        "adopt_challenger": adopt,
        "chosen_projection": CHALLENGER_COLUMN if adopt else PRIMARY_BENCHMARK,
    }


def _pair_key(pair: tuple[int, int]) -> str:
    return f"{pair[0]}->{pair[1]}"


def evaluate_development(
    ledgers: dict[int, pd.DataFrame], *, bootstrap_reps: int = BOOTSTRAP_REPS
) -> dict[str, Any]:
    """Run the whole development stage on season -> walk-forward ledger."""
    assert_projection_development_seasons(list(ledgers))
    for season, ledger in ledgers.items():
        assert_projection_development_seasons(sorted(int(s) for s in ledger["season"].unique()))
        if set(ledger["season"].astype(int).unique()) != {int(season)}:
            raise ValueError(f"Ledger keyed {season} carries other seasons")

    inputs = {season: build_season_inputs(ledgers[season]) for season in sorted(ledgers)}

    graded_by_pair: dict[tuple[int, int], pd.DataFrame] = {}
    pairs: dict[str, Any] = {}
    for pair in DEVELOPMENT_PAIRS:
        input_season, target_season = pair
        tables = build_pair(inputs[input_season], inputs[target_season])
        realized_fit, deserved_fit = fit_input_season_shrinkage(
            ledgers[input_season], target_season=target_season
        )
        graded = add_ladder(tables.graded, realized_fit=realized_fit, deserved_fit=deserved_fit)
        graded_by_pair[pair] = graded
        pairs[_pair_key(pair)] = {
            "input_season": input_season,
            "target_season": target_season,
            "survivorship": tables.survivorship,
            "shrinkage": {
                "realized": shrinkage_fit_record(realized_fit),
                "deserved": shrinkage_fit_record(deserved_fit),
            },
            "ladder_metrics": [
                m.as_record()
                for m in evaluate_ladder(graded, target=TARGET_COLUMN, predictors=LADDER)
            ],
            "reading": "descriptive (S6): no decision rests on these rungs",
        }

    # S5: fit on 2022->2023 only, evaluate once on 2023->2024.
    train = graded_by_pair[CHALLENGER_FIT_PAIR]
    cv = select_alpha_grouped_cv(train)
    fit = fit_challenger(
        train, alpha=cv["selected_alpha"], evaluation_input_season=ADOPTION_EVALUATION_PAIR[0]
    )
    evaluation = graded_by_pair[ADOPTION_EVALUATION_PAIR].copy()
    evaluation[CHALLENGER_COLUMN], n_fallback = predict_with_fallback(
        fit, evaluation, fallback_column=PRIMARY_BENCHMARK
    )

    adoption_record = pairs[_pair_key(ADOPTION_EVALUATION_PAIR)]
    adoption_record["ladder_metrics"].extend(
        m.as_record()
        for m in evaluate_ladder(evaluation, target=TARGET_COLUMN, predictors=(CHALLENGER_COLUMN,))
    )

    design = BootstrapDesign(reps=bootstrap_reps, seed=BOOTSTRAP_SEED, alpha=BOOTSTRAP_ALPHA)
    paired = evaluation.assign(season=ADOPTION_EVALUATION_PAIR[1], cutoff=0, horizon=0)
    bootstrap = run_paired_bootstrap(
        paired,
        target=TARGET_COLUMN,
        reference=PRIMARY_BENCHMARK,
        challenger=CHALLENGER_COLUMN,
        design=design,
    )
    cells = cast(dict[str, Any], bootstrap["cells"])
    delta = cells[_CELL]["pooled"]

    return {
        "specification": "docs/plans/2027_projection_model_plan.md S1-S8 (frozen 2026-10-08)",
        "pairs": pairs,
        "challenger": {
            "fit_pair": _pair_key(CHALLENGER_FIT_PAIR),
            "alpha_selection": cv,
            "fit": fit.as_record(),
            "n_evaluation_rows_using_fallback": n_fallback,
        },
        "bootstrap": {
            "reps": design.reps,
            "seed": design.seed,
            "alpha": design.alpha,
            "cluster_column": design.cluster_column,
            "n_clusters": bootstrap["n_clusters_resampled"],
            "sign_convention": "delta = metric(challenger) - metric(shrunk deserved); negative favours the challenger",
            "delta_mae": delta["delta_mae"],
            "delta_rmse": delta["delta_rmse"],
        },
        "adoption": adoption_decision(delta["delta_mae"]),
        "disclosures": [DESIGN_TIME_DISCLOSURE],
    }
