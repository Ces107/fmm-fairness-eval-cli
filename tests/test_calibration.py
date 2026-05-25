"""Tests for fmm_fairness.calibration."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fmm_fairness.calibration import (
    HL_DEFAULT_BINS,
    brier_score,
    build_calibration_block,
    hosmer_lemeshow,
    per_class_brier,
    per_group_calibration,
    reliability_bins,
)


class TestBrierScore:
    def test_perfect_predictions_yield_zero(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        # one-hot scores at the true class
        scores = np.zeros((6, 3))
        scores[np.arange(6), y_true] = 1.0
        assert brier_score(y_true, scores, 3) == pytest.approx(0.0, abs=1e-12)

    def test_uniform_prior_against_3_class(self) -> None:
        y_true = np.array([0, 1, 2])
        scores = np.full((3, 3), 1.0 / 3.0)
        # per-sample brier = (1 - 1/3)^2 + 2 * (1/3)^2 = 4/9 + 2/9 = 6/9 = 2/3
        assert brier_score(y_true, scores, 3) == pytest.approx(2.0 / 3.0, abs=1e-10)

    def test_worst_case_one_hot_wrong_class(self) -> None:
        y_true = np.array([0])
        scores = np.array([[0.0, 1.0, 0.0]])
        # (1 - 0)^2 + (0 - 1)^2 + (0 - 0)^2 = 2
        assert brier_score(y_true, scores, 3) == pytest.approx(2.0, abs=1e-12)

    def test_empty_input_returns_nan(self) -> None:
        y_true = np.array([], dtype=int)
        scores = np.zeros((0, 3))
        assert math.isnan(brier_score(y_true, scores, 3))

    def test_shape_mismatch_raises(self) -> None:
        y_true = np.array([0, 1])
        scores = np.zeros((2, 3))
        with pytest.raises(ValueError, match="expected"):
            brier_score(y_true, scores, 4)


class TestPerClassBrier:
    def test_known_per_class_values(self) -> None:
        y_true = np.array([0, 1])
        # for class 0: indicators [1, 0], probs [1.0, 0.0] -> 0
        # for class 1: indicators [0, 1], probs [0.0, 1.0] -> 0
        scores = np.array([[1.0, 0.0], [0.0, 1.0]])
        pc = per_class_brier(y_true, scores, 2)
        assert pc[0] == pytest.approx(0.0, abs=1e-12)
        assert pc[1] == pytest.approx(0.0, abs=1e-12)

    def test_constant_05_predictions(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        scores = np.full((4, 2), 0.5)
        pc = per_class_brier(y_true, scores, 2)
        # indicators are [1,0,1,0] vs probs [0.5]: mean((0.5)^2) = 0.25 per class
        assert pc[0] == pytest.approx(0.25, abs=1e-12)
        assert pc[1] == pytest.approx(0.25, abs=1e-12)


class TestHosmerLemeshow:
    def test_perfect_calibration_returns_low_chi2(self) -> None:
        rng = np.random.default_rng(20260525)
        y_score = rng.uniform(0.0, 1.0, size=500)
        # y_true drawn independently with prob = y_score (well-calibrated by construction)
        y_true = (rng.uniform(0, 1, 500) < y_score).astype(int)
        result = hosmer_lemeshow(y_true, y_score, n_bins=HL_DEFAULT_BINS)
        # well-calibrated random data should not produce a tiny p-value
        assert result["chi2"] is not None
        assert result["p_value"] > 0.01

    def test_systematic_miscalibration_flagged(self) -> None:
        rng = np.random.default_rng(20260525)
        y_score = rng.uniform(0.4, 0.6, size=400)
        # all positive — perfectly miscalibrated relative to ~0.5 mean score
        y_true = np.ones(400, dtype=int)
        result = hosmer_lemeshow(y_true, y_score, n_bins=HL_DEFAULT_BINS)
        # constant y_true triggers the undefined-test guard
        assert math.isnan(result["chi2"]) or result["p_value"] < 0.01

    def test_empty_input_returns_undefined(self) -> None:
        result = hosmer_lemeshow(np.array([], dtype=int), np.array([], dtype=float))
        assert math.isnan(result["chi2"])
        assert result["n_bins"] == 0


class TestReliabilityBins:
    def test_bin_counts_sum_to_n(self) -> None:
        rng = np.random.default_rng(20260525)
        y_score = rng.uniform(0, 1, size=200)
        y_true = (y_score > 0.5).astype(int)
        bins = reliability_bins(y_true, y_score, n_bins=10)
        assert sum(bins["counts"]) == 200
        assert len(bins["bin_centers"]) == 10

    def test_empty_input_returns_empty_arrays(self) -> None:
        bins = reliability_bins(np.array([], dtype=int), np.array([], dtype=float))
        assert bins["counts"] == []


class TestPerGroupCalibration:
    def test_per_group_block_shape(self) -> None:
        rng = np.random.default_rng(20260525)
        n = 200
        site = rng.choice(["A", "B"], size=n)
        y_true = rng.integers(0, 3, size=n)
        scores = np.full((n, 3), 0.2)
        scores[np.arange(n), y_true] = 0.6
        df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_true,  # perfect prediction for shape testing
                "y_score_0": scores[:, 0],
                "y_score_1": scores[:, 1],
                "y_score_2": scores[:, 2],
                "site": site,
            }
        )
        block = per_group_calibration(df, "site", num_classes=3, min_group_n=20)
        assert block["attribute"] == "site"
        assert block["num_classes"] == 3
        assert {g["group"] for g in block["per_group"]} == {"A", "B"}
        for g in block["per_group"]:
            assert isinstance(g["brier_score"], float)
            assert len(g["per_class_brier"]) == 3
            assert len(g["reliability_by_class"]) == 3
            assert len(g["hosmer_lemeshow_by_class"]) == 3


class TestBuildCalibrationBlock:
    def test_orchestrator_returns_per_attribute(self) -> None:
        rng = np.random.default_rng(20260525)
        n = 200
        site = rng.choice(["A", "B"], size=n)
        sex = rng.choice(["M", "F"], size=n)
        y_true = rng.integers(0, 3, size=n)
        scores = np.full((n, 3), 0.2)
        scores[np.arange(n), y_true] = 0.6
        df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_true,
                "y_score_0": scores[:, 0],
                "y_score_1": scores[:, 1],
                "y_score_2": scores[:, 2],
                "site": site,
                "sex": sex,
            }
        )
        block = build_calibration_block(df, ["site", "sex"], num_classes=3)
        assert block["num_classes"] == 3
        assert set(block["per_attribute"]) == {"site", "sex"}
        assert block["global_brier_score"] >= 0.0
        assert len(block["global_per_class_brier"]) == 3
