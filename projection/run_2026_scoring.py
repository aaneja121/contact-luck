"""2027 projection: score 2026 ONCE as input (step 6 of the plan, S8).

    .venv/bin/python -m projection.run_2026_scoring

Reads the development result to learn which rung ships (S6), pins the
archived 2026-09-27 v1.1 snapshot by label, verifies it against its own
integrity hashes, and projects it onto the contact stage through the existing
read-only adapter (`forecast.phase2.snapshot`). Shrinkage is fit on 2026
inputs only (S4 option a). Every ladder rung is written for every projected
batter (S1: at least one resolved 2026 batted ball), with the chosen rung
copied to `projection_2027`.

Writes `outputs/projection_2027/projections/projections_2027.csv` and a
manifest recording its SHA-256. Committing that hash before 2027-03-24 is
what binds the A3 grade; this module does not commit, upload or publish.

Guards, in order, all before 2026 is read: a clean working tree; no existing
output (one look); an output path outside the sealed and prospective
namespaces; a development result that chose a rung this module can produce.
Then the snapshot must reach 2026-09-27 and be integrity-verified, and the
ledger must be 2026 only. No 2027 data is read. Offline.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from forecast.baselines import shrinkage_fit_record
from forecast.forecast_config import CHRONOLOGICAL_SORT_KEY
from projection.ladder import add_ladder, fit_input_season_shrinkage
from projection.projection_config import (
    DEVELOPMENT_RESULTS_PATH,
    LADDER,
    PRIMARY_BENCHMARK,
    PROJECTION_INPUT_DATA_THROUGH,
    PROJECTION_INPUT_SEASON,
    PROJECTION_INPUT_SNAPSHOT_LABEL,
    PROJECTION_TEST_SEASON,
    PROJECTIONS_OUTPUT_DIR,
    SPEC_PATH,
    assert_path_outside_forbidden_namespaces,
    assert_projection_input_season,
)
from projection.run_development import (
    _json_default,
    file_sha256,
    git_commit,
    working_tree_status,
)
from projection.season_table import build_season_inputs

logger = logging.getLogger(__name__)

PROJECTIONS_FILENAME = "projections_2027.csv"
MANIFEST_FILENAME = "projections_2027_manifest.json"
PROJECTION_COLUMN = "projection_2027"

#: (ledger, snapshot record, contact-stage audit)
SnapshotLoader = Callable[[], tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]]


class ProjectionScoringError(RuntimeError):
    """Raised when a guard refuses the 2026 scoring run."""


def chosen_projection_from(development_results: dict[str, Any]) -> str:
    """The rung S6 chose. Only shrunk deserved is producible here.

    If the challenger had been adopted, C7 would require the frozen 2022->2023
    fit applied to 2026 inputs. That path was not built because the
    development stage did not adopt it; refusing beats improvising it.
    """
    adoption = development_results["adoption"]
    if adoption["adopt_challenger"] or adoption["chosen_projection"] != PRIMARY_BENCHMARK:
        raise ProjectionScoringError(
            "The development result adopted the challenger; this entry point only produces "
            f"{PRIMARY_BENCHMARK}. Build the C7 path (frozen 2022->2023 fit) as a reviewed change."
        )
    return PRIMARY_BENCHMARK


def assert_snapshot_is_season_end(record: dict[str, Any]) -> None:
    """The pinned snapshot must be integrity-verified and reach the last regular-season day."""
    if not record.get("integrity_verified"):
        raise ProjectionScoringError("The pinned snapshot was not integrity-verified.")
    if str(record.get("data_through_date")) != PROJECTION_INPUT_DATA_THROUGH:
        raise ProjectionScoringError(
            f"The projection input must be data through {PROJECTION_INPUT_DATA_THROUGH}; the "
            f"pinned snapshot runs through {record.get('data_through_date')}."
        )
    assert_projection_input_season([int(record["prospective_season"])])


def build_2027_projections(
    ledger: pd.DataFrame, *, chosen_projection: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every ladder rung for every batter with a resolved 2026 batted ball."""
    assert_projection_input_season(sorted(int(s) for s in ledger["season"].unique()))
    ordered = ledger.sort_values(
        ["batter", *CHRONOLOGICAL_SORT_KEY, "event_id"], kind="mergesort"
    ).reset_index(drop=True)

    inputs = build_season_inputs(ordered)
    realized_fit, deserved_fit = fit_input_season_shrinkage(
        ordered, target_season=PROJECTION_TEST_SEASON
    )
    laddered = add_ladder(inputs, realized_fit=realized_fit, deserved_fit=deserved_fit)

    table = pd.DataFrame(
        {
            "batter": laddered["batter"].astype("int64"),
            "input_season": PROJECTION_INPUT_SEASON,
            "projection_season": PROJECTION_TEST_SEASON,
            "n_resolved_bbe_2026": laddered["n_resolved_bbe_through_cutoff"].astype("int64"),
            "realized_rv_per_100_2026": laddered["std_realized_rv_per_100"].astype(float),
            "deserved_rv_per_100_2026": laddered["std_deserved_rv_per_100"].astype(float),
            **{rung: laddered[rung].astype(float) for rung in LADDER},
        }
    )
    table[PROJECTION_COLUMN] = table[chosen_projection]
    table = table.sort_values("batter", kind="mergesort").reset_index(drop=True)
    fits = {
        "realized": shrinkage_fit_record(realized_fit),
        "deserved": shrinkage_fit_record(deserved_fit),
    }
    return table, fits


def load_pinned_2026_snapshot() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Pin the 2026-09-27 snapshot, verify it, and project it onto the contact stage."""
    from forecast.phase2.snapshot import load_contact_stage_ledger, resolve_snapshot

    snapshot = resolve_snapshot(label=PROJECTION_INPUT_SNAPSHOT_LABEL)
    ledger, audit = load_contact_stage_ledger(snapshot)
    return ledger, snapshot.as_record(), audit


def run(
    *,
    load_snapshot: SnapshotLoader = load_pinned_2026_snapshot,
    development_results_path: Path = DEVELOPMENT_RESULTS_PATH,
    output_dir: Path = PROJECTIONS_OUTPUT_DIR,
) -> Path:
    """Run every guard, score 2026 once, and return the manifest path."""
    status = working_tree_status()
    if status:
        raise ProjectionScoringError(f"Working tree is not clean:\n{status}")
    projections_path = Path(output_dir) / PROJECTIONS_FILENAME
    manifest_path = Path(output_dir) / MANIFEST_FILENAME
    for path in (projections_path, manifest_path):
        if path.exists():
            raise ProjectionScoringError(
                f"{path} already exists. 2026 is scored once; a rerun needs a recorded "
                "defect fix, and any change after the hash is committed voids A3."
            )
    assert_path_outside_forbidden_namespaces(Path(output_dir))
    if not Path(development_results_path).exists():
        raise ProjectionScoringError(
            f"No development result at {development_results_path}; the development stage "
            "must run (and choose a rung) before 2026 is scored."
        )
    development_results = json.loads(Path(development_results_path).read_text())
    chosen = chosen_projection_from(development_results)

    ledger, snapshot_record, audit = load_snapshot()
    assert_snapshot_is_season_end(snapshot_record)
    table, fits = build_2027_projections(ledger, chosen_projection=chosen)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(projections_path, index=False, lineterminator="\n")
    manifest = {
        "specification": "docs/plans/2027_projection_model_plan.md S1-S8 (frozen 2026-10-08)",
        "chosen_projection": chosen,
        "projection_column": PROJECTION_COLUMN,
        "units": "contact-stage run value per 100 resolved batted balls",
        "projections_file": PROJECTIONS_FILENAME,
        "projections_sha256": file_sha256(projections_path),
        "n_projected": len(table),
        "n_projected_by_bbe": {
            "lt_50": int((table["n_resolved_bbe_2026"] < 50).sum()),
            "50_to_99": int(table["n_resolved_bbe_2026"].between(50, 99).sum()),
            "ge_100": int((table["n_resolved_bbe_2026"] >= 100).sum()),
        },
        "shrinkage": fits,
        "snapshot": snapshot_record,
        "contact_stage_audit": audit,
        "provenance": {
            "git_commit": git_commit(),
            "run_at_utc": datetime.now(UTC).isoformat(),
            "specification_sha256": file_sha256(SPEC_PATH),
            "development_results_sha256": file_sha256(Path(development_results_path)),
            "development_results_git_commit": development_results["provenance"]["git_commit"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    logger.info("wrote %s and %s", projections_path, manifest_path)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
