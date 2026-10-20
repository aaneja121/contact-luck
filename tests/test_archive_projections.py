"""Tests for scripts/archive_projections.py -- the S8 off-machine copy of the
frozen 2027 projections.

Every test uses `InMemoryArchiveClient` (from `scripts.archive_snapshot`);
none contact real Cloudflare R2, and none read the real gitignored
projections files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import archive_projections as ap
import ensure_frozen_inputs as efi
import pytest
from archive_snapshot import InMemoryArchiveClient

from projection import projection_config


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _specs_for(tmp_path: Path, contents: dict[str, bytes]) -> tuple[ap.ProjectionFileSpec, ...]:
    specs = []
    for name, data in contents.items():
        path = tmp_path / name
        path.write_bytes(data)
        specs.append(
            ap.ProjectionFileSpec(
                local_path=path, sha256=_sha256(data), r2_key=f"{ap.R2_PREFIX}{name}"
            )
        )
    return tuple(specs)


class TestPinnedFiles:
    def test_pins_both_frozen_files_under_the_projection_prefix(self) -> None:
        names = {spec.local_path.name for spec in ap.PROJECTION_FILES}
        assert names == {"projections_2027.csv", "projections_2027_manifest.json"}
        for spec in ap.PROJECTION_FILES:
            assert spec.r2_key == f"{ap.R2_PREFIX}{spec.local_path.name}"

    def test_prefix_is_separate_from_every_other_archive_prefix(self) -> None:
        assert ap.R2_PREFIX == "projection-2027/"
        assert not ap.R2_PREFIX.startswith(("prospective/", "frozen-inputs/"))

    def test_local_paths_match_the_projection_output_dir(self) -> None:
        for spec in ap.PROJECTION_FILES:
            assert spec.local_path.parent == projection_config.PROJECTIONS_OUTPUT_DIR

    def test_pinned_hashes_are_the_ones_committed_in_the_plan(self) -> None:
        plan = projection_config.SPEC_PATH.read_text(encoding="utf-8")
        for spec in ap.PROJECTION_FILES:
            assert f"`{spec.local_path.name}` | `{spec.sha256}`" in plan


class TestArchiveProjections:
    def test_uploads_every_file_and_verifies_it(self, tmp_path: Path) -> None:
        specs = _specs_for(tmp_path, {"a.csv": b"a,b\n1,2\n", "m.json": b"{}"})
        client = InMemoryArchiveClient()

        results = ap.archive_projections(client=client, specs=specs)

        assert [outcome for _, outcome in results] == [efi.FrozenInputUploadOutcome.UPLOADED] * 2
        for spec in specs:
            assert client.get_object_bytes(spec.r2_key) == spec.local_path.read_bytes()

    def test_rerun_over_identical_objects_is_a_no_op(self, tmp_path: Path) -> None:
        specs = _specs_for(tmp_path, {"a.csv": b"x"})
        client = InMemoryArchiveClient()
        ap.archive_projections(client=client, specs=specs)

        results = ap.archive_projections(client=client, specs=specs)

        assert results == [(specs[0].r2_key, efi.FrozenInputUploadOutcome.NO_OP_IDENTICAL)]

    def test_refuses_a_local_file_that_no_longer_matches_its_pin(self, tmp_path: Path) -> None:
        specs = _specs_for(tmp_path, {"a.csv": b"x"})
        specs[0].local_path.write_bytes(b"edited")
        client = InMemoryArchiveClient()

        with pytest.raises(efi.FrozenInputLocalHashMismatchError):
            ap.archive_projections(client=client, specs=specs)
        assert client.list_keys(ap.R2_PREFIX) == []

    def test_checks_every_local_file_before_uploading_any(self, tmp_path: Path) -> None:
        specs = _specs_for(tmp_path, {"a.csv": b"x", "m.json": b"y"})
        specs[1].local_path.write_bytes(b"edited")
        client = InMemoryArchiveClient()

        with pytest.raises(efi.FrozenInputLocalHashMismatchError):
            ap.archive_projections(client=client, specs=specs)
        assert client.list_keys(ap.R2_PREFIX) == []

    def test_never_overwrites_a_different_existing_object(self, tmp_path: Path) -> None:
        specs = _specs_for(tmp_path, {"a.csv": b"x"})
        client = InMemoryArchiveClient()
        client.put_object_bytes(specs[0].r2_key, b"something else")

        with pytest.raises(efi.FrozenInputUploadConflictError):
            ap.archive_projections(client=client, specs=specs)
        assert client.get_object_bytes(specs[0].r2_key) == b"something else"


class TestMain:
    def test_missing_credentials_exit_non_zero_without_uploading(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        for name in ("R2_BUCKET_NAME", "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(name, raising=False)

        assert ap.main([]) == 2
        assert "R2_BUCKET_NAME" in capsys.readouterr().err
