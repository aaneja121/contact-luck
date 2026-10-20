"""The S8 off-machine copy of the frozen 2027 projections.

`docs/plans/2027_projection_model_plan.md` (S8, decision 6) keeps a copy of
the frozen projections in the existing R2 archive bucket, because the A3
grade is impossible if the only copy is lost. The files themselves stay in
the gitignored `outputs/projection_2027/projections/`; the SHA-256 committed
in the plan is what binds the grade, and this module pins the same hashes
(a structural test checks they match the plan).

Every object lives under the `projection-2027/` prefix, separate from
`prospective/<season>/` (`scripts.archive_snapshot`) and `frozen-inputs/`
(`scripts.ensure_frozen_inputs`). Each file goes through
`ensure_frozen_inputs.upload_frozen_input_to_r2`, so the write-once rules are
the same: refuse a local file that doesn't match its pin, never overwrite a
different existing object, no-op on an identical one, and re-download every
upload to verify its hash.

ONE-TIME action, run by hand only when the maintainer asks for it. It reads
the four `R2_*` variables `scripts.archive_snapshot` uses and never prints
their values.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from archive_snapshot import ArchiveClient, make_r2_client
from ensure_frozen_inputs import (
    FrozenInputError,
    FrozenInputLocalHashMismatchError,
    FrozenInputUploadOutcome,
    upload_frozen_input_to_r2,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECTIONS_DIR = PROJECT_ROOT / "outputs" / "projection_2027" / "projections"
R2_PREFIX = "projection-2027/"


@dataclass(frozen=True)
class ProjectionFileSpec:
    local_path: Path
    sha256: str
    r2_key: str


def _spec(name: str, sha256: str) -> ProjectionFileSpec:
    return ProjectionFileSpec(
        local_path=PROJECTIONS_DIR / name, sha256=sha256, r2_key=f"{R2_PREFIX}{name}"
    )


#: The hashes committed in the plan's "2027 projections -- frozen 2026-10-13"
#: section (commit c804038). Never edit these: a changed file voids A3.
PROJECTION_FILES: tuple[ProjectionFileSpec, ...] = (
    _spec(
        "projections_2027.csv",
        "5cd8cf488c67e7994740c365773a0d15867a13aa1f17f99e9baf5bb0b3ae50b0",
    ),
    _spec(
        "projections_2027_manifest.json",
        "dc613249b239376a2d314b3a70aed5db24764f1f0c9719f9e9146e43f63797bb",
    ),
)


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_projections(
    *, client: ArchiveClient, specs: Sequence[ProjectionFileSpec] = PROJECTION_FILES
) -> list[tuple[str, FrozenInputUploadOutcome]]:
    """Upload every pinned file, write-once and verified. Every local file is
    checked against its pin before anything is uploaded, so a drifted file
    never leaves a partial copy behind.
    """
    for spec in specs:
        if not spec.local_path.exists():
            raise FrozenInputError(f"{spec.local_path} does not exist -- nothing to upload.")
        actual = _sha256_of_file(spec.local_path)
        if actual != spec.sha256:
            raise FrozenInputLocalHashMismatchError(
                f"Refusing to upload anything: {spec.local_path} has sha256 {actual}, not "
                f"the pinned {spec.sha256}. The frozen projections must never change."
            )
    return [
        (
            spec.r2_key,
            upload_frozen_input_to_r2(
                local_path=spec.local_path,
                expected_sha256=spec.sha256,
                r2_key=spec.r2_key,
                client=client,
            ),
        )
        for spec in specs
    ]


def _client_from_env() -> ArchiveClient:
    """Same four variables as `scripts.archive_snapshot`; duplicated rather
    than importing either module's private helper (the precedent set in
    `scripts.ensure_frozen_inputs`).
    """
    values = {
        name: os.environ.get(name)
        for name in ("R2_BUCKET_NAME", "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise FrozenInputError(
            f"Missing required environment variable(s) for R2 access: {', '.join(missing)}"
        )
    return make_r2_client(
        bucket=values["R2_BUCKET_NAME"] or "",
        account_id=values["R2_ACCOUNT_ID"] or "",
        access_key_id=values["R2_ACCESS_KEY_ID"] or "",
        secret_access_key=values["R2_SECRET_ACCESS_KEY"] or "",
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        client = _client_from_env()
        for r2_key, outcome in archive_projections(client=client):
            print(f"upload outcome={outcome.value} key={r2_key}")
        return 0
    except FrozenInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
