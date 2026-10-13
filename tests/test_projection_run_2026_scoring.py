"""2027 projection: the one-shot 2026 scoring entry point (synthetic, offline).

Nothing here reads a real 2026 snapshot. The snapshot loader is replaced by a
synthetic 2026 ledger in the play-ledger contact-stage shape.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tests.test_projection_development import _skills, _synthetic_ledger

from projection import run_2026_scoring
from projection.projection_config import (
    PROJECT_ROOT,
    ProjectionNamespaceError,
    ProjectionSeasonError,
    assert_projection_input_season,
)


def _development_results(*, adopt: bool = False) -> dict[str, Any]:
    chosen = "challenger_d_season" if adopt else "shrunk_deserved_persistence"
    return {
        "adoption": {"adopt_challenger": adopt, "chosen_projection": chosen},
        "provenance": {"git_commit": "a" * 40},
    }


@pytest.fixture()
def development_results_path(tmp_path: Path) -> Path:
    path = tmp_path / "development_results.json"
    path.write_text(json.dumps(_development_results()))
    return path


def _snapshot_record(data_through: str = "2026-09-27") -> dict[str, Any]:
    return {
        "snapshot_label": "2026-09-27",
        "data_through_date": data_through,
        "prospective_season": 2026,
        "integrity_verified": True,
        "integrity_hashes_verified": {"outputs/play_ledger.parquet": "b" * 64},
    }


def _loader(ledger: pd.DataFrame, record: dict[str, Any] | None = None) -> Any:
    def load() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
        return ledger, record or _snapshot_record(), {"contact_stage": True}

    return load


@pytest.fixture(scope="module")
def ledger_2026() -> pd.DataFrame:
    return _synthetic_ledger(2026, skill=_skills(n=60), seed=5, unresolved_share=0.05)


def _clean(monkeypatch: pytest.MonkeyPatch, clean: bool = True) -> None:
    monkeypatch.setattr(run_2026_scoring, "working_tree_status", lambda: "" if clean else " M x.py")
    monkeypatch.setattr(run_2026_scoring, "git_commit", lambda: "f" * 40)


# --- season guard ----------------------------------------------------------


def test_input_season_guard_allows_only_2026() -> None:
    assert_projection_input_season([2026])
    for season in (2022, 2024, 2025, 2027):
        with pytest.raises(ProjectionSeasonError):
            assert_projection_input_season([season])


# --- the projection table --------------------------------------------------


def test_projections_cover_every_batter_with_a_resolved_2026_bbe(
    ledger_2026: pd.DataFrame,
) -> None:
    table, fits = run_2026_scoring.build_2027_projections(
        ledger_2026, chosen_projection="shrunk_deserved_persistence"
    )
    resolved = ledger_2026[ledger_2026["outcome_class"].notna()]
    assert set(table["batter"]) == set(resolved["batter"])
    assert table["batter"].is_unique
    assert list(table["batter"]) == sorted(table["batter"])
    assert (table["n_resolved_bbe_2026"] >= 1).all()
    assert (table["input_season"] == 2026).all()
    assert (table["projection_season"] == 2027).all()
    assert fits["deserved"]["fit_seasons"] == [2026]
    assert fits["realized"]["fit_seasons"] == [2026]


def test_projection_equals_the_chosen_rung(ledger_2026: pd.DataFrame) -> None:
    table, _ = run_2026_scoring.build_2027_projections(
        ledger_2026, chosen_projection="shrunk_deserved_persistence"
    )
    pd.testing.assert_series_equal(
        table["projection_2027"],
        table["shrunk_deserved_persistence"],
        check_names=False,
    )
    for rung in ("league_mean", "shrunk_realized_persistence", "shrunk_deserved_persistence"):
        assert table[rung].notna().all()


def test_shrunk_deserved_lies_between_the_raw_rate_and_the_prior(
    ledger_2026: pd.DataFrame,
) -> None:
    table, fits = run_2026_scoring.build_2027_projections(
        ledger_2026, chosen_projection="shrunk_deserved_persistence"
    )
    mu = fits["deserved"]["prior_mean_per_100"]
    raw = table["deserved_rv_per_100_2026"]
    shrunk = table["shrunk_deserved_persistence"]
    low = pd.concat([raw, pd.Series(mu, index=raw.index)], axis=1).min(axis=1)
    high = pd.concat([raw, pd.Series(mu, index=raw.index)], axis=1).max(axis=1)
    assert ((shrunk >= low - 1e-9) & (shrunk <= high + 1e-9)).all()


def test_projections_refuse_a_ledger_that_is_not_2026(ledger_2026: pd.DataFrame) -> None:
    for season in (2025, 2027):
        with pytest.raises(ProjectionSeasonError):
            run_2026_scoring.build_2027_projections(
                ledger_2026.assign(season=season),
                chosen_projection="shrunk_deserved_persistence",
            )


def test_projections_are_deterministic(ledger_2026: pd.DataFrame) -> None:
    first, _ = run_2026_scoring.build_2027_projections(
        ledger_2026, chosen_projection="shrunk_deserved_persistence"
    )
    second, _ = run_2026_scoring.build_2027_projections(
        ledger_2026.sample(frac=1.0, random_state=1),
        chosen_projection="shrunk_deserved_persistence",
    )
    pd.testing.assert_frame_equal(first, second)


# --- development decision and snapshot gate --------------------------------


def test_chosen_projection_comes_from_the_development_result() -> None:
    assert (
        run_2026_scoring.chosen_projection_from(_development_results())
        == "shrunk_deserved_persistence"
    )


def test_an_adopted_challenger_is_refused_rather_than_improvised() -> None:
    with pytest.raises(run_2026_scoring.ProjectionScoringError, match="challenger"):
        run_2026_scoring.chosen_projection_from(_development_results(adopt=True))


def test_snapshot_must_reach_the_last_regular_season_day() -> None:
    run_2026_scoring.assert_snapshot_is_season_end(_snapshot_record())
    with pytest.raises(run_2026_scoring.ProjectionScoringError, match="2026-09-27"):
        run_2026_scoring.assert_snapshot_is_season_end(_snapshot_record("2026-09-01"))
    with pytest.raises(run_2026_scoring.ProjectionScoringError, match="integrity"):
        run_2026_scoring.assert_snapshot_is_season_end(
            {**_snapshot_record(), "integrity_verified": False}
        )


# --- the one-shot runner ---------------------------------------------------


def test_runner_writes_projections_and_a_manifest_that_binds_them(
    ledger_2026: pd.DataFrame,
    development_results_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean(monkeypatch)
    out = tmp_path / "out"
    manifest_path = run_2026_scoring.run(
        load_snapshot=_loader(ledger_2026),
        development_results_path=development_results_path,
        output_dir=out,
    )

    manifest = json.loads(manifest_path.read_text())
    projections = out / run_2026_scoring.PROJECTIONS_FILENAME
    assert manifest["projections_sha256"] == hashlib.sha256(projections.read_bytes()).hexdigest()
    assert manifest["chosen_projection"] == "shrunk_deserved_persistence"
    assert manifest["n_projected"] == len(pd.read_csv(projections))
    provenance = manifest["provenance"]
    assert provenance["git_commit"] == "f" * 40
    assert len(provenance["specification_sha256"]) == 64
    assert (
        provenance["development_results_sha256"]
        == hashlib.sha256(development_results_path.read_bytes()).hexdigest()
    )
    assert manifest["snapshot"]["data_through_date"] == "2026-09-27"


def test_runner_refuses_a_second_run(
    ledger_2026: pd.DataFrame,
    development_results_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean(monkeypatch)
    kwargs = {
        "load_snapshot": _loader(ledger_2026),
        "development_results_path": development_results_path,
        "output_dir": tmp_path / "out",
    }
    run_2026_scoring.run(**kwargs)
    with pytest.raises(run_2026_scoring.ProjectionScoringError, match="already exists"):
        run_2026_scoring.run(**kwargs)


def test_runner_refuses_a_dirty_tree_before_reading_2026(
    development_results_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean(monkeypatch, clean=False)

    def must_not_load() -> Any:
        raise AssertionError("2026 was read before the clean-tree guard")

    with pytest.raises(run_2026_scoring.ProjectionScoringError, match="not clean"):
        run_2026_scoring.run(
            load_snapshot=must_not_load,
            development_results_path=development_results_path,
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_runner_refuses_a_missing_development_result_before_reading_2026(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean(monkeypatch)

    def must_not_load() -> Any:
        raise AssertionError("2026 was read before the development result was checked")

    with pytest.raises(run_2026_scoring.ProjectionScoringError, match="development"):
        run_2026_scoring.run(
            load_snapshot=must_not_load,
            development_results_path=tmp_path / "missing.json",
            output_dir=tmp_path / "out",
        )


def test_runner_refuses_an_early_snapshot(
    ledger_2026: pd.DataFrame,
    development_results_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean(monkeypatch)
    with pytest.raises(run_2026_scoring.ProjectionScoringError, match="2026-09-27"):
        run_2026_scoring.run(
            load_snapshot=_loader(ledger_2026, _snapshot_record("2026-09-01")),
            development_results_path=development_results_path,
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_runner_refuses_a_prospective_output_namespace(
    ledger_2026: pd.DataFrame,
    development_results_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean(monkeypatch)
    with pytest.raises(ProjectionNamespaceError):
        run_2026_scoring.run(
            load_snapshot=_loader(ledger_2026),
            development_results_path=development_results_path,
            output_dir=PROJECT_ROOT / "outputs" / "prospective" / "v1_1" / "x",
        )
