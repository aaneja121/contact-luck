"""Tests for the 2027 projection grader (A3; plan step 7, C14-C19).

Synthetic data only. Nothing here reads real 2027 data, the real frozen
projections, the v1.3 namespace, or the network.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP
from projection import grade_2027 as g
from projection import run_2027_grading as r
from projection.projection_config import (
    LADDER,
    PRIMARY_BENCHMARK,
    REALIZED_BENCHMARK,
    SPEC_PATH,
    TARGET_COLUMN,
)

# --------------------------------------------------------------------------
# Synthetic builders
# --------------------------------------------------------------------------


def _cleaned(rows: list[tuple[int, bool, str | None]], season: int = 2027) -> pd.DataFrame:
    """(batter, is_eligible, outcome_class) per pitch-level row."""
    return pd.DataFrame(
        {
            "season": season,
            "batter": [b for b, _, _ in rows],
            "is_eligible": [e for _, e, _ in rows],
            "outcome_class": pd.array([o for _, _, o in rows], dtype="string"),
        }
    )


def _cleaned_for(outcomes_by_batter: dict[int, list[str]]) -> pd.DataFrame:
    rows = [(b, True, o) for b, outs in outcomes_by_batter.items() for o in outs]
    return _cleaned(rows)


def _projections(batters: list[int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    deserved = rng.normal(4.0, 2.0, len(batters))
    frame = pd.DataFrame(
        {
            "batter": batters,
            "league_mean": 4.0,
            "shrunk_realized_persistence": deserved + rng.normal(0, 1.5, len(batters)),
            "shrunk_deserved_persistence": deserved,
        }
    )
    frame["projection_2027"] = frame[PRIMARY_BENCHMARK]
    return frame


def _targets(rates: dict[int, float], n: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batter": list(rates),
            "n_resolved_bbe_through_cutoff": float(n),
            "std_realized_rv_per_100": list(rates.values()),
        }
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# The 2027 target (C15)
# --------------------------------------------------------------------------


class TestRealizedTargets:
    def test_rate_is_100_times_mean_run_value_over_resolved_balls(self) -> None:
        cleaned = _cleaned_for({1: ["single", "out", "out", "home_run"]})
        targets, _ = g.realized_targets(cleaned)
        expected = 100 * np.mean(
            [DEFAULT_RUN_VALUE_MAP[o] for o in ("single", "out", "out", "home_run")]
        )
        row = targets.set_index("batter").loc[1]
        assert row["std_realized_rv_per_100"] == pytest.approx(expected)
        assert row["n_resolved_bbe_through_cutoff"] == 4

    def test_ineligible_and_unresolved_rows_leave_the_denominator(self) -> None:
        cleaned = _cleaned(
            [
                (1, True, "single"),
                (1, True, None),  # field_error / fielders_choice: unresolved
                (1, False, None),  # not a fair batted ball
                (1, False, "out"),  # ineligible even with a label
            ]
        )
        targets, audit = g.realized_targets(cleaned)
        assert targets.set_index("batter").loc[1, "n_resolved_bbe_through_cutoff"] == 1
        assert audit == {
            "n_input_rows": 4,
            "n_eligible": 2,
            "n_resolved": 1,
            "n_eligible_unresolved": 1,
            "n_batters_with_resolved_bbe": 1,
        }

    def test_refuses_any_season_other_than_2027(self) -> None:
        with pytest.raises(g.GradingError, match="2027"):
            g.realized_targets(_cleaned([(1, True, "out")], season=2026))

    def test_refuses_mixed_seasons(self) -> None:
        mixed = pd.concat([_cleaned([(1, True, "out")]), _cleaned([(1, True, "out")], season=2025)])
        with pytest.raises(g.GradingError):
            g.realized_targets(mixed)

    def test_refuses_an_unknown_outcome_class(self) -> None:
        with pytest.raises(g.GradingError, match="run-value"):
            g.realized_targets(_cleaned([(1, True, "not_a_class")]))


# --------------------------------------------------------------------------
# Grading (S1, S7, C16-C18)
# --------------------------------------------------------------------------


class TestGrade:
    def _graded_case(self, *, deserved_better: bool) -> dict[str, Any]:
        batters = list(range(1, 81))
        projections = _projections(batters)
        rng = np.random.default_rng(1)
        anchor = (
            projections[PRIMARY_BENCHMARK] if deserved_better else projections[REALIZED_BENCHMARK]
        )
        rates = dict(zip(batters, anchor + rng.normal(0, 0.3, len(batters)), strict=True))
        return g.grade(
            projections,
            _targets(rates),
            chosen_projection=PRIMARY_BENCHMARK,
            bootstrap_reps=200,
        )

    def test_primary_is_chosen_minus_shrunk_realized(self) -> None:
        result = self._graded_case(deserved_better=True)
        primary = result["primary"]
        assert primary["reference"] == REALIZED_BENCHMARK
        assert primary["challenger"] == "projection_2027"
        assert primary["delta_mae"]["point_estimate"] < 0
        assert result["classification"]["classification"] == "established_incremental_success"

    def test_realized_better_is_classified_against(self) -> None:
        result = self._graded_case(deserved_better=False)
        assert result["primary"]["delta_mae"]["point_estimate"] > 0
        assert result["classification"]["classification"] == "evidence_against_incremental_value"

    def test_classification_uses_mae_only_and_carries_no_2026_wording(self) -> None:
        classification = self._graded_case(deserved_better=True)["classification"]
        assert classification["deciding_metric"] == "MAE"
        assert "meaning" not in classification
        assert "2026" not in json.dumps(classification)

    def test_secondary_metrics_are_reported_without_a_decision(self) -> None:
        result = self._graded_case(deserved_better=True)
        assert result["secondary"]["vs_league_mean"]["reference"] == "league_mean"
        predictors = [m["predictor"] for m in result["secondary"]["ladder_metrics"]]
        assert predictors == list(LADDER)
        for metric in result["secondary"]["ladder_metrics"]:
            assert {"mae", "rmse", "pearson", "spearman"} <= set(metric)

    def test_bootstrap_uses_the_frozen_convention(self) -> None:
        bootstrap = self._graded_case(deserved_better=True)["bootstrap"]
        assert bootstrap["seed"] == 42
        assert bootstrap["alpha"] == 0.05
        assert bootstrap["cluster_column"] == "batter"

    def test_survivorship_counts_projected_graded_and_new_hitters(self) -> None:
        projections = _projections([1, 2, 3, 4])
        targets = pd.concat(
            [
                _targets({1: 3.0, 2: 5.0}, n=150),
                _targets({3: 4.0}, n=99),  # below the grading minimum
                _targets({9: 6.0}, n=200),  # qualified, no 2026 row
            ]
        )
        result = g.grade(
            pd.concat([projections] * 1),
            targets,
            chosen_projection=PRIMARY_BENCHMARK,
            bootstrap_reps=50,
        )
        assert result["survivorship"] == {
            "n_projected": 4,
            "n_projected_with_any_target_bbe": 3,
            "n_graded": 2,
            "n_projected_not_graded": 2,
            "n_target_qualified_without_input_row": 1,
        }

    def test_refuses_when_projection_column_is_not_the_chosen_rung(self) -> None:
        projections = _projections([1, 2, 3])
        projections["projection_2027"] = projections[REALIZED_BENCHMARK]
        with pytest.raises(g.GradingError, match="chosen"):
            g.grade(
                projections,
                _targets({1: 1.0, 2: 2.0, 3: 3.0}),
                chosen_projection=PRIMARY_BENCHMARK,
            )

    def test_refuses_when_nobody_is_graded(self) -> None:
        with pytest.raises(g.GradingError, match="graded"):
            g.grade(
                _projections([1, 2]),
                _targets({1: 1.0, 2: 2.0}, n=10),
                chosen_projection=PRIMARY_BENCHMARK,
            )

    def test_target_column_is_the_frozen_name(self) -> None:
        assert g.grade.__module__ == "projection.grade_2027"
        assert TARGET_COLUMN == "target_realized_rv_per_100"


# --------------------------------------------------------------------------
# Binding to the frozen projections and to v1.3's raw file (C14)
# --------------------------------------------------------------------------


class TestPins:
    def test_pins_are_the_hashes_committed_in_the_plan(self) -> None:
        plan = SPEC_PATH.read_text(encoding="utf-8")
        for name, sha in g.PINNED_PROJECTION_SHA256.items():
            assert f"`{name}` | `{sha}`" in plan

    def test_matching_files_verify(self, tmp_path: Path) -> None:
        (tmp_path / "a.csv").write_text("x")
        pins = {"a.csv": _sha(tmp_path / "a.csv")}
        assert g.verify_pinned_projections(tmp_path, pins=pins) == pins

    def test_a_changed_file_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "a.csv").write_text("x")
        with pytest.raises(g.GradingError, match="voids A3"):
            g.verify_pinned_projections(tmp_path, pins={"a.csv": "0" * 64})

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(g.GradingError, match="R2"):
            g.verify_pinned_projections(tmp_path, pins={"a.csv": "0" * 64})


class TestV13RawFile:
    def _record(self, raw: Path, sha: str) -> dict[str, Any]:
        return {"ingestion": {"statcast": {"path": str(raw), "sha256": sha}}}

    def test_matching_raw_file_verifies(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw.parquet"
        raw.write_bytes(b"raw")
        assert g.verify_raw_matches_v13_record(self._record(raw, _sha(raw)), raw)["sha256"] == (
            _sha(raw)
        )

    def test_a_different_raw_file_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw.parquet"
        raw.write_bytes(b"raw")
        with pytest.raises(g.GradingError, match="SHA-256"):
            g.verify_raw_matches_v13_record(self._record(raw, "0" * 64), raw)

    def test_a_different_path_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw.parquet"
        raw.write_bytes(b"raw")
        other = tmp_path / "other.parquet"
        with pytest.raises(g.GradingError, match="path"):
            g.verify_raw_matches_v13_record(self._record(other, _sha(raw)), raw)

    def test_a_record_without_statcast_provenance_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(g.GradingError):
            g.verify_raw_matches_v13_record({"ingestion": {}}, tmp_path / "raw.parquet")


# --------------------------------------------------------------------------
# The entry point's guards (C19)
# --------------------------------------------------------------------------

_END = date(2027, 9, 26)


class _World:
    """A synthetic world: frozen projections, a v1.3 record and raw file, a cleaned 2027."""

    def __init__(self, tmp_path: Path) -> None:
        self.projections_dir = tmp_path / "projections"
        self.projections_dir.mkdir()
        batters = list(range(1, 41))
        _projections(batters).to_csv(self.projections_dir / g.PROJECTIONS_FILENAME, index=False)
        (self.projections_dir / g.MANIFEST_FILENAME).write_text(
            json.dumps({"chosen_projection": PRIMARY_BENCHMARK})
        )
        self.pins = {
            name: _sha(self.projections_dir / name)
            for name in (g.PROJECTIONS_FILENAME, g.MANIFEST_FILENAME)
        }
        self.raw_path = tmp_path / "v1_3" / "raw.parquet"
        self.raw_path.parent.mkdir()
        self.raw_path.write_bytes(b"synthetic raw 2027")
        self.comparison_path = tmp_path / "v1_3" / "comparison_2027.json"
        self.comparison_path.write_text(
            json.dumps(
                {
                    "guards": {"season_end_date": _END.isoformat()},
                    "ingestion": {
                        "statcast": {"path": str(self.raw_path), "sha256": _sha(self.raw_path)}
                    },
                }
            )
        )
        self.output_dir = tmp_path / "grade"
        self.cleaned = _cleaned_for({b: ["single", "out", "out"] * 40 for b in batters})

    def kwargs(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "tree_status": lambda: "",
            "season_end": (_END, "official schedule citation", "2027-10-01"),
            "comparison_path": self.comparison_path,
            "raw_path": self.raw_path,
            "projections_dir": self.projections_dir,
            "output_dir": self.output_dir,
            "pins": self.pins,
            "check_schedule": lambda start, end: {"n_final": 2430, "end": end.isoformat()},
            "load_cleaned": lambda: self.cleaned,
            "bootstrap_reps": 50,
        }
        base.update(overrides)
        return base


class TestRunGuards:
    def test_end_to_end_writes_one_grade(self, tmp_path: Path) -> None:
        world = _World(tmp_path)
        path = r.run(**world.kwargs())
        record = json.loads(path.read_text())
        assert record["grading"]["survivorship"]["n_graded"] == 40
        assert record["provenance"]["v1_3_raw"]["sha256"] == _sha(world.raw_path)
        assert record["provenance"]["projections"] == world.pins
        assert record["guards"]["season_end_date"] == _END.isoformat()

    def test_second_run_is_refused(self, tmp_path: Path) -> None:
        world = _World(tmp_path)
        r.run(**world.kwargs())
        with pytest.raises(r.GradingRunError, match="once"):
            r.run(**world.kwargs())

    def test_dirty_tree_is_refused_before_anything_is_read(self, tmp_path: Path) -> None:
        world = _World(tmp_path)

        def _never() -> pd.DataFrame:
            raise AssertionError("2027 was read")

        with pytest.raises(r.GradingRunError, match="clean"):
            r.run(**world.kwargs(tree_status=lambda: " M x.py", load_cleaned=_never))

    def test_unrecorded_season_end_is_refused(self, tmp_path: Path) -> None:
        world = _World(tmp_path)
        with pytest.raises(r.GradingRunError, match="end date"):
            r.run(**world.kwargs(season_end=(None, None, None)))

    def test_refuses_before_the_v13_comparison_has_run(self, tmp_path: Path) -> None:
        world = _World(tmp_path)
        world.comparison_path.unlink()
        with pytest.raises(r.GradingRunError, match="v1.3"):
            r.run(**world.kwargs())

    def test_refuses_when_v13_recorded_a_different_season_end(self, tmp_path: Path) -> None:
        world = _World(tmp_path)
        with pytest.raises(r.GradingRunError, match="season end"):
            r.run(**world.kwargs(season_end=(date(2027, 9, 27), "citation", "2027-10-01")))

    def test_incomplete_season_is_refused(self, tmp_path: Path) -> None:
        world = _World(tmp_path)

        def _incomplete(start: date, end: date) -> dict[str, Any]:
            raise ValueError("3 game(s) not final")

        with pytest.raises(ValueError, match="not final"):
            r.run(**world.kwargs(check_schedule=_incomplete))
        assert not world.output_dir.exists()

    def test_changed_projections_are_refused(self, tmp_path: Path) -> None:
        world = _World(tmp_path)
        (world.projections_dir / g.PROJECTIONS_FILENAME).write_text("batter\n1\n")
        with pytest.raises(g.GradingError, match="voids A3"):
            r.run(**world.kwargs())
        assert not world.output_dir.exists()

    def test_output_inside_a_forbidden_namespace_is_refused(self, tmp_path: Path) -> None:
        world = _World(tmp_path)
        forbidden = r.PROJECT_ROOT / "data" / "final_evaluation" / "2025" / "grade"
        with pytest.raises(Exception, match="forbidden"):
            r.run(**world.kwargs(output_dir=forbidden))

    def test_default_output_is_its_own_namespace(self) -> None:
        assert r.GRADE_OUTPUT_DIR == r.PROJECT_ROOT / "outputs" / "projection_2027" / "grade"
