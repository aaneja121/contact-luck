"""2027 projection, development stage: synthetic-data tests only.

Nothing here reads a real ledger, a 2026 snapshot, or any 2027 data, and
nothing touches the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projection.challenger import (
    fit_challenger,
    predict_with_fallback,
    select_alpha_grouped_cv,
)
from projection.development import (
    adoption_decision,
    build_pair,
    evaluate_development,
)
from projection.ladder import add_ladder, fit_input_season_shrinkage
from projection.projection_config import (
    ADOPTION_EVALUATION_PAIR,
    CHALLENGER_COLUMN,
    CHALLENGER_FIT_PAIR,
    CHALLENGER_NOMINAL_FEATURES,
    GRADING_MIN_RESOLVED_BBE,
    LADDER,
    PRIMARY_BENCHMARK,
    ProjectionSeasonError,
    assert_projection_development_seasons,
)
from projection.season_table import ProjectionInputError, build_season_inputs

# --------------------------------------------------------------------------
# Synthetic ledger
# --------------------------------------------------------------------------

_CLASSES = ("out", "single", "double", "triple", "home_run")
_VALUES = {"out": -0.27, "single": 0.47, "double": 0.77, "triple": 1.04, "home_run": 1.40}


def _synthetic_ledger(
    season: int,
    *,
    n_batters: int = 120,
    seed: int = 0,
    skill: dict[int, float] | None = None,
    unresolved_share: float = 0.0,
) -> pd.DataFrame:
    """Per-batted-ball rows in the walk-forward ledger's shape.

    Each batter has a latent `skill` that shifts both his deserved value and
    his outcome odds, so the ladder has something real to find.
    """
    rng = np.random.default_rng(seed + season)
    skill = skill or {b: float(rng.normal(0.0, 0.05)) for b in range(1, n_batters + 1)}
    rows: list[dict[str, object]] = []
    event = 0
    for batter, ability in skill.items():
        n = int(rng.integers(20, 420))
        for i in range(n):
            event += 1
            expected = float(np.clip(0.02 + ability + rng.normal(0.0, 0.08), -0.25, 0.9))
            p_hit = float(np.clip(0.30 + 2.0 * ability, 0.05, 0.9))
            klass = _CLASSES[0] if rng.random() > p_hit else str(rng.choice(_CLASSES[1:]))
            observed = _VALUES[klass]
            unresolved = rng.random() < unresolved_share
            rows.append(
                {
                    "event_id": f"{season}-{event}",
                    "game_pk": 1000 + i,
                    "at_bat_number": 1 + (i % 5),
                    "pitch_number": 1,
                    "game_date": pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=i // 3),
                    "season": season,
                    "batter": batter,
                    "stand": "L" if batter % 3 == 0 else "R",
                    "launch_speed": float(88.0 + 40.0 * ability + rng.normal(0.0, 12.0)),
                    "launch_angle": float(rng.normal(12.0, 25.0)),
                    "spray_angle_approx": float(rng.normal(0.0, 25.0)),
                    "bb_type": str(rng.choice(["ground_ball", "line_drive", "fly_ball", "popup"])),
                    "outcome_class": None if unresolved else klass,
                    "observed_contact_result_run_value": np.nan if unresolved else observed,
                    "baseline_expected_contact_run_value": expected,
                    "contact_result_surprise": np.nan if unresolved else observed - expected,
                }
            )
    return pd.DataFrame(rows)


def _skills(n: int = 160, seed: int = 7) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    return {b: float(rng.normal(0.0, 0.05)) for b in range(1, n + 1)}


@pytest.fixture(scope="module")
def dev_ledgers() -> dict[int, pd.DataFrame]:
    skills = _skills()
    return {s: _synthetic_ledger(s, skill=skills, seed=11) for s in (2022, 2023, 2024)}


# --------------------------------------------------------------------------
# Season guards
# --------------------------------------------------------------------------


def test_development_seasons_are_allowed() -> None:
    assert_projection_development_seasons([2022, 2023, 2024])


@pytest.mark.parametrize(
    ("season", "fragment"),
    [(2025, "sealed"), (2026, "input season"), (2027, "test season"), (2021, "not a")],
)
def test_other_seasons_are_refused(season: int, fragment: str) -> None:
    with pytest.raises(ProjectionSeasonError, match=fragment):
        assert_projection_development_seasons([2023, season])


def test_pairs_are_the_two_walk_forward_pairs() -> None:
    assert CHALLENGER_FIT_PAIR == (2022, 2023)
    assert ADOPTION_EVALUATION_PAIR == (2023, 2024)


def test_challenger_features_drop_only_the_within_season_recent_window() -> None:
    from forecast.ridge import FEATURE_SETS

    full = FEATURE_SETS["D_full_contact_profile"]
    assert all(not f.startswith("recent_") for f in CHALLENGER_NOMINAL_FEATURES)
    assert set(full) - set(CHALLENGER_NOMINAL_FEATURES) == {
        f for f in full if f.startswith("recent_")
    }


# --------------------------------------------------------------------------
# Season inputs
# --------------------------------------------------------------------------


def test_season_inputs_are_resolved_bbe_rates() -> None:
    ledger = _synthetic_ledger(2023, n_batters=10, unresolved_share=0.1)
    table = build_season_inputs(ledger)

    resolved = ledger[ledger["outcome_class"].notna()]
    expected = resolved.groupby("batter").agg(
        n=("observed_contact_result_run_value", "size"),
        realized=("observed_contact_result_run_value", "mean"),
        deserved=("baseline_expected_contact_run_value", "mean"),
    )
    table = table.set_index("batter")
    assert table.index.is_unique
    assert (table["season"] == 2023).all()
    np.testing.assert_array_equal(
        table.loc[expected.index, "n_resolved_bbe_through_cutoff"], expected["n"]
    )
    np.testing.assert_allclose(
        table.loc[expected.index, "std_realized_rv_per_100"], 100 * expected["realized"]
    )
    np.testing.assert_allclose(
        table.loc[expected.index, "std_deserved_rv_per_100"], 100 * expected["deserved"]
    )
    assert set(CHALLENGER_NOMINAL_FEATURES) <= set(table.columns)


def test_season_inputs_refuse_a_broken_contact_identity() -> None:
    ledger = _synthetic_ledger(2023, n_batters=5)
    ledger.loc[ledger.index[0], "contact_result_surprise"] += 0.5
    with pytest.raises(ProjectionInputError, match="identity"):
        build_season_inputs(ledger)


def test_season_inputs_refuse_more_than_one_season() -> None:
    both = pd.concat([_synthetic_ledger(2022, n_batters=5), _synthetic_ledger(2023, n_batters=5)])
    with pytest.raises(ProjectionInputError, match="one season"):
        build_season_inputs(both)


# --------------------------------------------------------------------------
# Pairs: population and survivorship (S1)
# --------------------------------------------------------------------------


def test_pair_population_projected_graded_and_unprojected(
    dev_ledgers: dict[int, pd.DataFrame],
) -> None:
    inputs = build_season_inputs(dev_ledgers[2022])
    targets = build_season_inputs(dev_ledgers[2023])
    # A hitter who appears only in the target season (a "rookie").
    targets = pd.concat(
        [targets, targets.iloc[[0]].assign(batter=99999, n_resolved_bbe_through_cutoff=150.0)],
        ignore_index=True,
    )
    pair = build_pair(inputs, targets)

    assert len(pair.projected) == len(inputs)
    graded = pair.graded
    assert (graded["target_n_resolved_bbe"] >= GRADING_MIN_RESOLVED_BBE).all()
    assert set(graded["batter"]) <= set(inputs["batter"])
    assert pair.survivorship["n_target_qualified_without_input_row"] == 1
    assert 99999 not in set(graded["batter"])
    assert pair.survivorship["n_projected"] == len(inputs)
    assert pair.survivorship["n_graded"] == len(graded)


# --------------------------------------------------------------------------
# Ladder (S3, S4 option a)
# --------------------------------------------------------------------------


def test_shrinkage_is_fit_on_the_input_season_only(
    dev_ledgers: dict[int, pd.DataFrame],
) -> None:
    realized, deserved = fit_input_season_shrinkage(dev_ledgers[2023], target_season=2024)
    assert realized.fit_seasons == (2023,)
    assert deserved.fit_seasons == (2023,)


def test_shrinkage_refuses_events_from_the_target_season(
    dev_ledgers: dict[int, pd.DataFrame],
) -> None:
    with pytest.raises(ProjectionInputError, match="input season"):
        fit_input_season_shrinkage(dev_ledgers[2024], target_season=2024)
    both = pd.concat([dev_ledgers[2022], dev_ledgers[2023]])
    with pytest.raises(ProjectionInputError, match="input season"):
        fit_input_season_shrinkage(both, target_season=2024)


def test_ladder_rungs(dev_ledgers: dict[int, pd.DataFrame]) -> None:
    inputs = build_season_inputs(dev_ledgers[2023])
    realized, deserved = fit_input_season_shrinkage(dev_ledgers[2023], target_season=2024)
    out = add_ladder(inputs, realized_fit=realized, deserved_fit=deserved)

    assert set(LADDER) <= set(out.columns)
    assert out["league_mean"].nunique() == 1
    assert out["league_mean"].iloc[0] == pytest.approx(realized.prior_mean)
    # Shrinkage moves every rate toward (never past) its prior mean.
    for rung, raw, fit in (
        ("shrunk_realized_persistence", "std_realized_rv_per_100", realized),
        ("shrunk_deserved_persistence", "std_deserved_rv_per_100", deserved),
    ):
        gap_raw = (out[raw] - fit.prior_mean).abs()
        gap_shrunk = (out[rung] - fit.prior_mean).abs()
        assert (gap_shrunk <= gap_raw + 1e-12).all()


# --------------------------------------------------------------------------
# Challenger (S5)
# --------------------------------------------------------------------------


def _graded_pair(ledgers: dict[int, pd.DataFrame], pair: tuple[int, int]) -> pd.DataFrame:
    return build_pair(
        build_season_inputs(ledgers[pair[0]]), build_season_inputs(ledgers[pair[1]])
    ).graded


def test_alpha_cv_never_splits_a_batter(dev_ledgers: dict[int, pd.DataFrame]) -> None:
    train = _graded_pair(dev_ledgers, CHALLENGER_FIT_PAIR)
    record = select_alpha_grouped_cv(train)
    assert record["n_folds"] == 5
    for fold in record["folds"]:
        assert not set(fold["train_batters"]) & set(fold["validation_batters"])
    assert record["selected_alpha"] in record["alpha_grid"]
    assert record["train_seasons"] == [2022]


def test_challenger_refuses_training_rows_from_the_evaluation_input_season(
    dev_ledgers: dict[int, pd.DataFrame],
) -> None:
    train = _graded_pair(dev_ledgers, ADOPTION_EVALUATION_PAIR)
    with pytest.raises(ValueError, match="strictly earlier"):
        fit_challenger(train, alpha=1.0, evaluation_input_season=2023)


def test_challenger_falls_back_to_the_benchmark_for_incomplete_features(
    dev_ledgers: dict[int, pd.DataFrame],
) -> None:
    train = _graded_pair(dev_ledgers, CHALLENGER_FIT_PAIR)
    fit = fit_challenger(train, alpha=1.0, evaluation_input_season=2023)

    frame = _graded_pair(dev_ledgers, ADOPTION_EVALUATION_PAIR)
    realized, deserved = fit_input_season_shrinkage(dev_ledgers[2023], target_season=2024)
    frame = add_ladder(frame, realized_fit=realized, deserved_fit=deserved)
    frame.loc[frame.index[:3], "launch_speed_sd"] = np.nan

    values, n_fallback = predict_with_fallback(fit, frame, fallback_column=PRIMARY_BENCHMARK)
    assert n_fallback == 3
    np.testing.assert_allclose(values.iloc[:3], frame[PRIMARY_BENCHMARK].iloc[:3])
    assert np.isfinite(values).all()


def test_challenger_training_drops_incomplete_rows_and_counts_them(
    dev_ledgers: dict[int, pd.DataFrame],
) -> None:
    train = _graded_pair(dev_ledgers, CHALLENGER_FIT_PAIR)
    train.loc[train.index[:2], "spray_angle_sd"] = np.nan
    fit = fit_challenger(train, alpha=1.0, evaluation_input_season=2023)
    assert fit.n_train == len(train) - 2


# --------------------------------------------------------------------------
# Adoption rule (S6) and the development run
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("point", "upper", "adopt"),
    [(-0.2, -0.01, True), (-0.2, 0.05, False), (0.1, 0.3, False), (-0.2, 0.0, False)],
)
def test_adoption_requires_a_negative_point_and_an_interval_entirely_below_zero(
    point: float, upper: float, adopt: bool
) -> None:
    delta = {"point_estimate": point, "ci_lower": point - 0.3, "ci_upper": upper}
    decision = adoption_decision(delta)
    assert decision["adopt_challenger"] is adopt
    assert decision["chosen_projection"] == (CHALLENGER_COLUMN if adopt else PRIMARY_BENCHMARK)


def test_development_run_is_deterministic_and_complete(
    dev_ledgers: dict[int, pd.DataFrame],
) -> None:
    first = evaluate_development(dev_ledgers, bootstrap_reps=50)
    second = evaluate_development(dev_ledgers, bootstrap_reps=50)
    assert first == second

    assert set(first["pairs"]) == {"2022->2023", "2023->2024"}
    for record in first["pairs"].values():
        rungs = {row["predictor"] for row in record["ladder_metrics"]}
        assert set(LADDER) <= rungs
        assert record["shrinkage"]["realized"]["fit_seasons"] == [record["input_season"]]
    adoption = first["pairs"]["2023->2024"]
    assert CHALLENGER_COLUMN in {row["predictor"] for row in adoption["ladder_metrics"]}
    assert first["adoption"]["rule"].startswith("adopt the challenger only if")
    assert first["adoption"]["chosen_projection"] in (CHALLENGER_COLUMN, PRIMARY_BENCHMARK)
    assert "2026 Contact Forecast result was known" in first["disclosures"][0]
    assert first["bootstrap"]["seed"] == 42


def test_development_refuses_a_non_development_season(
    dev_ledgers: dict[int, pd.DataFrame],
) -> None:
    ledgers = dict(dev_ledgers)
    ledgers[2025] = dev_ledgers[2024].assign(season=2025)
    with pytest.raises(ProjectionSeasonError):
        evaluate_development(ledgers, bootstrap_reps=10)
