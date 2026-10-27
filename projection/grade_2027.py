"""2027 projection: grade the frozen projections against 2027 (A3; plan step 7, S7).

Pure functions. `projection.run_2027_grading` owns the guards and the one
read of 2027; this module only checks inputs and computes the S7 record.

- The 2027 target (C15) needs no model: it is the frozen run-value table
  applied to each eligible batted ball's recorded outcome, exactly as the
  development ledgers define the realized rate
  (`forecast.score_development_seasons`). Resolved means eligible with a
  non-null `outcome_class`, so field errors and fielder's choices leave the
  denominator, matching production.
- The projections are read, never recomputed (C14). Their bytes must match
  the SHA-256 committed in the plan before 2027-03-24.
- S7: primary = delta_MAE (projection_2027 - shrunk realized) on graded
  hitters, batter-clustered bootstrap (500 reps, seed 42), classified by
  Contact Forecast's four-way rule on MAE only. Everything else is reported
  alongside it with no decision attached.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from forecast.bootstrap import BootstrapDesign, run_paired_bootstrap
from forecast.metrics import evaluate_ladder
from forecast.phase2.run_phase2_evaluation import classify_result
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP
from projection.development import build_pair
from projection.projection_config import (
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    LADDER,
    PROJECTION_TEST_SEASON,
    REALIZED_BENCHMARK,
    TARGET_COLUMN,
)

PROJECTIONS_FILENAME = "projections_2027.csv"
MANIFEST_FILENAME = "projections_2027_manifest.json"
PROJECTION_COLUMN = "projection_2027"

#: The hashes committed in the plan's "2027 projections -- frozen 2026-10-13"
#: section (commit c804038). A test checks they match the plan. Never edit.
PINNED_PROJECTION_SHA256: dict[str, str] = {
    PROJECTIONS_FILENAME: "5cd8cf488c67e7994740c365773a0d15867a13aa1f17f99e9baf5bb0b3ae50b0",
    MANIFEST_FILENAME: "dc613249b239376a2d314b3a70aed5db24764f1f0c9719f9e9146e43f63797bb",
}

#: The single bootstrap cell label: full 2026 season -> full 2027 season.
_CELL = "K0_H0"


class GradingError(RuntimeError):
    """Raised when an input cannot support the one permitted 2027 grade."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_projections(
    directory: Path, *, pins: Mapping[str, str] = PINNED_PROJECTION_SHA256
) -> dict[str, str]:
    """Every pinned file exists and hashes to its committed SHA-256."""
    verified: dict[str, str] = {}
    for name, expected in pins.items():
        path = Path(directory) / name
        if not path.exists():
            raise GradingError(
                f"{path} is missing. Restore it from the R2 copy (projection-2027/{name}) "
                "and check its hash; it can never be regenerated."
            )
        actual = file_sha256(path)
        if actual != expected:
            raise GradingError(
                f"{path} has SHA-256 {actual}, not the committed {expected}. Grading a "
                "changed projection voids A3."
            )
        verified[name] = actual
    return verified


def verify_raw_matches_v13_record(record: Mapping[str, Any], raw_path: Path) -> dict[str, str]:
    """A3: read only the raw 2027 file the v1.3 run downloaded, checked by hash."""
    try:
        statcast = record["ingestion"]["statcast"]
        recorded_path, recorded_sha = str(statcast["path"]), str(statcast["sha256"])
    except (KeyError, TypeError) as exc:
        raise GradingError(
            "The v1.3 comparison record has no ingestion.statcast path and SHA-256."
        ) from exc
    if Path(recorded_path).resolve() != Path(raw_path).resolve():
        raise GradingError(
            f"v1.3 recorded its raw file at {recorded_path}, but the grader was pointed at "
            f"{raw_path}; the path must be the same file."
        )
    if not Path(raw_path).exists():
        raise GradingError(f"{raw_path} is missing; the grader never downloads 2027 itself.")
    actual = file_sha256(Path(raw_path))
    if actual != recorded_sha:
        raise GradingError(
            f"{raw_path} has SHA-256 {actual}, not the {recorded_sha} the v1.3 run recorded."
        )
    return {"path": str(raw_path), "sha256": actual}


def realized_targets(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """C15: per-batter resolved-BBE count and realized rate per 100, 2027 only.

    Columns use the development target table's names (`build_pair` input).
    """
    seasons = sorted(int(s) for s in cleaned["season"].dropna().unique())
    if seasons != [PROJECTION_TEST_SEASON]:
        raise GradingError(f"The grader reads {PROJECTION_TEST_SEASON} only; got {seasons}.")

    eligible = cleaned["is_eligible"].fillna(False).astype(bool)
    resolved = eligible & cleaned["outcome_class"].notna()
    rows = cleaned.loc[resolved]
    values = rows["outcome_class"].astype(object).map(DEFAULT_RUN_VALUE_MAP)
    if values.isna().any():
        unknown = sorted(set(rows.loc[values.isna(), "outcome_class"].astype(str)))
        raise GradingError(f"No frozen run-value for outcome class(es) {unknown}.")

    per_batter = (
        pd.DataFrame({"batter": rows["batter"].astype("int64"), "rv": values.astype(float)})
        .groupby("batter", sort=True)["rv"]
        .agg(["size", "mean"])
        .reset_index()
    )
    targets = pd.DataFrame(
        {
            "batter": per_batter["batter"],
            "n_resolved_bbe_through_cutoff": per_batter["size"].astype(float),
            "std_realized_rv_per_100": 100.0 * per_batter["mean"],
        }
    )
    audit = {
        "n_input_rows": int(len(cleaned)),
        "n_eligible": int(eligible.sum()),
        "n_resolved": int(resolved.sum()),
        "n_eligible_unresolved": int((eligible & ~resolved).sum()),
        "n_batters_with_resolved_bbe": int(len(targets)),
    }
    return targets, audit


def _paired(frame: pd.DataFrame, *, reference: str, design: BootstrapDesign) -> dict[str, Any]:
    bootstrap = run_paired_bootstrap(
        frame,
        target=TARGET_COLUMN,
        reference=reference,
        challenger=PROJECTION_COLUMN,
        design=design,
    )
    cells = cast(dict[str, Any], bootstrap["cells"])
    pooled = cells[_CELL]["pooled"]
    return {
        "reference": reference,
        "challenger": PROJECTION_COLUMN,
        "sign_convention": f"delta = metric({PROJECTION_COLUMN}) - metric({reference}); "
        "negative favours the projection",
        "n_clusters": bootstrap["n_clusters_resampled"],
        "delta_mae": pooled["delta_mae"],
        "delta_rmse": pooled["delta_rmse"],
    }


def grade(
    projections: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    chosen_projection: str,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    """S1 and S7 on the frozen projections and the 2027 targets."""
    if not projections[PROJECTION_COLUMN].equals(projections[chosen_projection]):
        raise GradingError(
            f"{PROJECTION_COLUMN} is not the chosen rung {chosen_projection} the frozen "
            "manifest names."
        )

    tables = build_pair(projections, targets)
    graded = tables.graded
    if graded.empty:
        raise GradingError("No projected hitter reached the grading minimum; nothing is graded.")

    design = BootstrapDesign(reps=bootstrap_reps, seed=BOOTSTRAP_SEED, alpha=BOOTSTRAP_ALPHA)
    frame = graded.assign(season=PROJECTION_TEST_SEASON, cutoff=0, horizon=0)
    primary = _paired(frame, reference=REALIZED_BENCHMARK, design=design)
    delta = primary["delta_mae"]
    decided = classify_result(
        float(delta["point_estimate"]), float(delta["ci_lower"]), float(delta["ci_upper"])
    )

    return {
        "chosen_projection": chosen_projection,
        "units": "contact-stage run value per 100 resolved batted balls",
        "survivorship": tables.survivorship,
        "bootstrap": design.as_record(),
        "primary": primary,
        "classification": {
            "classification": decided["classification"],
            "condition": decided["condition"],
            "deciding_metric": "MAE",
            "applies_to": "primary only (S7)",
        },
        "secondary": {
            "reading": "reported together, no metric privileged, no decision rests on these",
            "vs_league_mean": _paired(frame, reference="league_mean", design=design),
            "ladder_metrics": [
                m.as_record()
                for m in evaluate_ladder(graded, target=TARGET_COLUMN, predictors=LADDER)
            ],
        },
    }
