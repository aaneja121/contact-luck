"""2027 projection: run the development stage ONCE (step 5 of the plan).

    .venv/bin/python -m projection.run_development

Reads the three walk-forward ledgers in `data/forecast/` (2022-2024), runs
`projection.development.evaluate_development`, and writes
`outputs/projection_2027/development/development_results.json`.

Guards, in order: a clean working tree (the recorded commit must be the code
that ran); no existing output (one look -- a rerun needs a recorded reason,
never a second try); ledger and output paths outside the sealed and
prospective namespaces. The ledgers' SHA-256, the commit and the frozen
specification's SHA-256 are recorded with the results. Offline: nothing here
downloads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projection.development import evaluate_development
from projection.projection_config import (
    BOOTSTRAP_REPS,
    DEVELOPMENT_LEDGER_DIR,
    DEVELOPMENT_OUTPUT_DIR,
    DEVELOPMENT_SEASONS,
    PROJECT_ROOT,
    SPEC_PATH,
    assert_path_outside_forbidden_namespaces,
    assert_projection_development_seasons,
)

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "development_results.json"


class DevelopmentRunError(RuntimeError):
    """Raised when a guard refuses the development run."""


def working_tree_status() -> str:
    """`git status --porcelain` output; empty means clean."""
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"{type(value)!r} is not JSON-serializable")


def run(
    *,
    ledger_dir: Path = DEVELOPMENT_LEDGER_DIR,
    output_dir: Path = DEVELOPMENT_OUTPUT_DIR,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> Path:
    """Run every guard, then the development stage; return the results path."""
    status = working_tree_status()
    if status:
        raise DevelopmentRunError(f"Working tree is not clean:\n{status}")
    results_path = Path(output_dir) / RESULTS_FILENAME
    if results_path.exists():
        raise DevelopmentRunError(
            f"{results_path} already exists. The development stage runs once; a rerun "
            "needs a recorded defect fix, not a second look."
        )
    assert_path_outside_forbidden_namespaces(Path(ledger_dir))
    assert_path_outside_forbidden_namespaces(Path(output_dir))
    assert_projection_development_seasons(list(DEVELOPMENT_SEASONS))

    paths = {
        season: Path(ledger_dir) / f"walk_forward_ledger_{season}.parquet"
        for season in DEVELOPMENT_SEASONS
    }
    ledgers = {season: pd.read_parquet(path) for season, path in paths.items()}
    results = evaluate_development(ledgers, bootstrap_reps=bootstrap_reps)
    results["provenance"] = {
        "git_commit": git_commit(),
        "run_at_utc": datetime.now(UTC).isoformat(),
        "specification_path": str(SPEC_PATH.relative_to(PROJECT_ROOT)),
        "specification_sha256": file_sha256(SPEC_PATH),
        "ledger_sha256": {str(s): file_sha256(p) for s, p in paths.items()},
        "bootstrap_reps": bootstrap_reps,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=_json_default))
    logger.info("wrote %s", results_path)
    return results_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
