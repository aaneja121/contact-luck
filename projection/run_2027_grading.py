"""2027 projection: grade the frozen projections ONCE (A3; plan step 7, C14-C19).

    PYTHONPATH=prospective .venv/bin/python -m projection.run_2027_grading

The only projection code that may read 2027 (RESEARCH_RULES.md, "A second 2027
reader IS authorized"). It never downloads: it reads the raw 2027 file the v1.3
comparison already downloaded, and only if that file's SHA-256 matches the v1.3
run's own record. So it runs after the v1.3 comparison, never before.

Guards, in order, all before any 2027 row is read:

1. a clean working tree;
2. no existing grade (one look);
3. an output path outside the sealed and prospective namespaces;
4. the 2027 season end recorded with its citation (`v1_3_2027_config`);
5. the v1.3 comparison has run and recorded that same season end;
6. every regular-season game through the end is final, postponed or cancelled,
   and none completed after it (`v1_3_2027_config.check_season_complete`);
7. the frozen projections match the SHA-256 committed in the plan;
8. the raw 2027 file matches the v1.3 record.

Then 2027 is cleaned exactly as v1.3 cleans it (`clean_development_data`) and
graded (`projection.grade_2027`). Writes one JSON, write-once. Publishes
nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import v1_3_2027_config as cfg

from forecast.forecast_config import PROJECT_ROOT
from mlb_luck_score.config import development_raw_path
from projection.grade_2027 import (
    MANIFEST_FILENAME,
    PINNED_PROJECTION_SHA256,
    PROJECTIONS_FILENAME,
    file_sha256,
    grade,
    realized_targets,
    verify_pinned_projections,
    verify_raw_matches_v13_record,
)
from projection.projection_config import (
    BOOTSTRAP_REPS,
    PROJECTION_OUTPUTS_DIR,
    PROJECTION_TEST_SEASON,
    PROJECTIONS_OUTPUT_DIR,
    SPEC_PATH,
    assert_path_outside_forbidden_namespaces,
)
from projection.run_development import _json_default, git_commit, working_tree_status

__all__ = ["PROJECT_ROOT"]

logger = logging.getLogger(__name__)

GRADE_OUTPUT_DIR = PROJECTION_OUTPUTS_DIR / "grade"
GRADE_FILENAME = "grade_2027.json"

#: The raw 2027 Statcast file the v1.3 run writes (`v1_3_2027_ingestion.ingest_2027`).
V1_3_RAW_PATH = development_raw_path(cfg.RAW_DIR, PROJECTION_TEST_SEASON)

#: (end date, official-schedule citation, verified-at), recorded after the season.
SeasonEnd = tuple[date | None, str | None, str | None]


class GradingRunError(RuntimeError):
    """Raised when a guard refuses the 2027 grading run."""


def _default_schedule_check(start: date, end: date) -> dict[str, Any]:
    return cfg.check_season_complete(start, end)


def _default_load_cleaned() -> pd.DataFrame:
    from mlb_luck_score.data.clean_development_data import clean_development_data

    return clean_development_data([PROJECTION_TEST_SEASON], cfg.RAW_DIR)


def run(
    *,
    tree_status: Callable[[], str] = working_tree_status,
    season_end: SeasonEnd = (
        cfg.SEASON_2027_END_DATE,
        cfg.SEASON_2027_END_SOURCE,
        cfg.SEASON_2027_END_VERIFIED_AT,
    ),
    comparison_path: Path = cfg.COMPARISON_PATH,
    raw_path: Path = V1_3_RAW_PATH,
    projections_dir: Path = PROJECTIONS_OUTPUT_DIR,
    output_dir: Path = GRADE_OUTPUT_DIR,
    pins: Mapping[str, str] = PINNED_PROJECTION_SHA256,
    check_schedule: Callable[[date, date], dict[str, Any]] = _default_schedule_check,
    load_cleaned: Callable[[], pd.DataFrame] = _default_load_cleaned,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> Path:
    """Run every guard, grade once, and return the grade path."""
    status = tree_status()
    if status:
        raise GradingRunError(f"Working tree is not clean:\n{status}")
    grade_path = Path(output_dir) / GRADE_FILENAME
    if grade_path.exists():
        raise GradingRunError(
            f"{grade_path} exists. The 2027 grade runs once; a second look needs the "
            "maintainer's explicit sign-off."
        )
    assert_path_outside_forbidden_namespaces(Path(output_dir))

    try:
        end = cfg.assert_season_end_recorded(*season_end)
    except cfg.SeasonNotRecordedError as exc:
        raise GradingRunError(f"The 2027 season end date is not recorded: {exc}") from exc
    if not Path(comparison_path).exists():
        raise GradingRunError(
            f"No v1.3 comparison at {comparison_path}. The grader reads the raw 2027 file "
            "the v1.3 run downloads, so v1.3 must run first."
        )
    v13_record = json.loads(Path(comparison_path).read_text())
    v13_end = v13_record.get("guards", {}).get("season_end_date")
    if v13_end != end.isoformat():
        raise GradingRunError(
            f"The v1.3 run recorded season end {v13_end}, but the config records "
            f"{end.isoformat()}; both must describe the same season."
        )
    schedule = check_schedule(cfg.SEASON_2027_START_DATE, end)
    projections_verified = verify_pinned_projections(Path(projections_dir), pins=pins)
    raw = verify_raw_matches_v13_record(v13_record, Path(raw_path))

    targets, target_audit = realized_targets(load_cleaned())
    projections = pd.read_csv(Path(projections_dir) / PROJECTIONS_FILENAME)
    manifest = json.loads((Path(projections_dir) / MANIFEST_FILENAME).read_text())
    result = grade(
        projections,
        targets,
        chosen_projection=manifest["chosen_projection"],
        bootstrap_reps=bootstrap_reps,
    )

    record = {
        "specification": "docs/plans/2027_projection_model_plan.md S1-S8 (frozen 2026-10-08)",
        "authorization": "RESEARCH_RULES.md, A3: a second 2027 reader, for grading only",
        "guards": {
            "season_start_date": cfg.SEASON_2027_START_DATE.isoformat(),
            "season_end_date": end.isoformat(),
            "season_end_source": season_end[1],
            "season_end_verified_at": season_end[2],
            "schedule": schedule,
        },
        "target_audit": target_audit,
        "grading": result,
        "provenance": {
            "git_commit": git_commit(),
            "run_at_utc": datetime.now(UTC).isoformat(),
            "specification_sha256": file_sha256(SPEC_PATH),
            "projections": projections_verified,
            "v1_3_raw": raw,
            "v1_3_comparison_sha256": file_sha256(Path(comparison_path)),
        },
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        with grade_path.open("x") as fh:
            json.dump(record, fh, indent=2, sort_keys=True, default=_json_default)
    except FileExistsError as exc:
        raise GradingRunError(f"{grade_path} appeared during the run; graded once only.") from exc
    logger.info("wrote %s: %s", grade_path, result["classification"]["classification"])
    return grade_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
