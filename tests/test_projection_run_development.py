"""2027 projection: the one-shot development runner's guards (synthetic, offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.test_projection_development import _skills, _synthetic_ledger

from projection import run_development
from projection.projection_config import PROJECT_ROOT, ProjectionNamespaceError


@pytest.fixture()
def ledger_dir(tmp_path: Path) -> Path:
    skills = _skills(n=80)
    for season in (2022, 2023, 2024):
        _synthetic_ledger(season, skill=skills, seed=3).to_parquet(
            tmp_path / f"walk_forward_ledger_{season}.parquet", index=False
        )
    return tmp_path


def _clean(monkeypatch: pytest.MonkeyPatch, clean: bool = True) -> None:
    monkeypatch.setattr(run_development, "working_tree_status", lambda: "" if clean else " M x.py")
    monkeypatch.setattr(run_development, "git_commit", lambda: "f" * 40)


def test_runner_writes_results_with_provenance(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean(monkeypatch)
    out = tmp_path / "out"
    path = run_development.run(ledger_dir=ledger_dir, output_dir=out, bootstrap_reps=20)

    results = json.loads(path.read_text())
    provenance = results["provenance"]
    assert provenance["git_commit"] == "f" * 40
    assert set(provenance["ledger_sha256"]) == {"2022", "2023", "2024"}
    assert len(provenance["specification_sha256"]) == 64
    assert results["adoption"]["chosen_projection"]


def test_runner_refuses_a_second_run(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean(monkeypatch)
    out = tmp_path / "out"
    run_development.run(ledger_dir=ledger_dir, output_dir=out, bootstrap_reps=20)
    with pytest.raises(run_development.DevelopmentRunError, match="already exists"):
        run_development.run(ledger_dir=ledger_dir, output_dir=out, bootstrap_reps=20)


def test_runner_refuses_a_dirty_tree(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean(monkeypatch, clean=False)
    with pytest.raises(run_development.DevelopmentRunError, match="not clean"):
        run_development.run(ledger_dir=ledger_dir, output_dir=tmp_path / "out", bootstrap_reps=20)
    assert not (tmp_path / "out").exists()


def test_runner_refuses_a_prospective_namespace(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean(monkeypatch)
    with pytest.raises(ProjectionNamespaceError):
        run_development.run(
            ledger_dir=PROJECT_ROOT / "data" / "prospective" / "2026",
            output_dir=ledger_dir / "out",
            bootstrap_reps=20,
        )


def test_runner_refuses_a_ledger_whose_rows_are_another_season(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean(monkeypatch)
    path = ledger_dir / "walk_forward_ledger_2024.parquet"
    pd.read_parquet(path).assign(season=2025).to_parquet(path, index=False)
    with pytest.raises(ValueError):
        run_development.run(ledger_dir=ledger_dir, output_dir=tmp_path / "out", bootstrap_reps=20)
