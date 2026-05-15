"""Tests for fmm_fairness.metrics with reproducible synthetic fixtures."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fmm_fairness.metrics import (
    calibration_gap,
    demographic_parity_gap,
    equal_opportunity_gap,
    inter_site_auc_variance,
    samd_fairness_score,
)

RNG_SEED = 1234


def _make_balanced_df(n: int = 400, sites: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    """Two sites, identical TPR/score distribution → gaps ≈ 0."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for site in sites:
        y_true = rng.integers(0, 2, size=n)
        y_score = np.where(y_true == 1,
                           rng.beta(8, 2, size=n),
                           rng.beta(2, 8, size=n))
        y_pred = (y_score >= 0.5).astype(int)
        for yt, ys, yp in zip(y_true, y_score, y_pred, strict=True):
            rows.append({"y_true": int(yt), "y_pred": int(yp), "y_score": float(ys), "site": site})
    return pd.DataFrame(rows)


def _make_biased_df(n: int = 400) -> pd.DataFrame:
    """Site A predicts well; site B predicts poorly (mostly negatives). Gap large."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    # Site A — strong model
    y_true_a = rng.integers(0, 2, size=n)
    y_score_a = np.where(y_true_a == 1,
                         rng.beta(9, 2, size=n),
                         rng.beta(2, 9, size=n))
    y_pred_a = (y_score_a >= 0.5).astype(int)
    for yt, ys, yp in zip(y_true_a, y_score_a, y_pred_a, strict=True):
        rows.append({"y_true": int(yt), "y_pred": int(yp), "y_score": float(ys), "site": "A"})
    # Site B — under-predicts positives (low TPR)
    y_true_b = rng.integers(0, 2, size=n)
    y_score_b = np.where(y_true_b == 1,
                         rng.beta(3, 5, size=n),  # positives barely above negatives
                         rng.beta(2, 9, size=n))
    y_pred_b = (y_score_b >= 0.5).astype(int)
    for yt, ys, yp in zip(y_true_b, y_score_b, y_pred_b, strict=True):
        rows.append({"y_true": int(yt), "y_pred": int(yp), "y_score": float(ys), "site": "B"})
    return pd.DataFrame(rows)


class TestEqualOpportunityGap(unittest.TestCase):
    def test_balanced_gap_is_small(self) -> None:
        df = _make_balanced_df()
        r = equal_opportunity_gap(df, "site", bootstrap_iters=100)
        self.assertEqual(r.metric_name, "equal_opportunity_gap")
        self.assertEqual(r.attribute, "site")
        self.assertLess(r.gap, 0.15)
        self.assertEqual(len(r.per_group), 2)

    def test_biased_gap_is_large(self) -> None:
        df = _make_biased_df()
        r = equal_opportunity_gap(df, "site", bootstrap_iters=100)
        self.assertGreater(r.gap, 0.20)
        # Boot CI present
        self.assertIsNotNone(r.gap_ci_low)
        self.assertIsNotNone(r.gap_ci_high)

    def test_excludes_small_groups(self) -> None:
        df = _make_balanced_df()
        # add 5-row "C" site → must be excluded under default min_group_n=20
        tiny = pd.DataFrame({
            "y_true": [1, 0, 1, 1, 0],
            "y_pred": [1, 0, 0, 1, 0],
            "y_score": [0.8, 0.1, 0.4, 0.9, 0.2],
            "site": ["C"] * 5,
        })
        df2 = pd.concat([df, tiny], ignore_index=True)
        with self.assertWarns(UserWarning):
            r = equal_opportunity_gap(df2, "site", bootstrap_iters=50)
        self.assertIn("C", r.excluded_groups)


class TestDemographicParityGap(unittest.TestCase):
    def test_balanced(self) -> None:
        df = _make_balanced_df()
        r = demographic_parity_gap(df, "site", bootstrap_iters=50)
        self.assertLess(r.gap, 0.15)

    def test_biased(self) -> None:
        df = _make_biased_df()
        r = demographic_parity_gap(df, "site", bootstrap_iters=50)
        self.assertGreater(r.gap, 0.10)


class TestCalibrationGap(unittest.TestCase):
    def test_calibration_is_nonnegative(self) -> None:
        df = _make_balanced_df()
        r = calibration_gap(df, "site", bootstrap_iters=50)
        self.assertGreaterEqual(r.gap, 0.0)
        for grp in r.per_group:
            self.assertGreaterEqual(grp.value, 0.0)


class TestInterSiteAUCVariance(unittest.TestCase):
    def test_balanced_variance_small(self) -> None:
        df = _make_balanced_df()
        r = inter_site_auc_variance(df, "site")
        self.assertLess(r.gap, 0.01)

    def test_biased_variance_larger(self) -> None:
        df = _make_biased_df()
        r_bal = inter_site_auc_variance(_make_balanced_df(), "site")
        r_bias = inter_site_auc_variance(df, "site")
        self.assertGreater(r_bias.gap, r_bal.gap)


class TestSamdFairnessScore(unittest.TestCase):
    def test_balanced_score_is_high(self) -> None:
        df = _make_balanced_df()
        # add a dummy demographic attribute that has no real bias
        df["sex"] = np.where(np.arange(len(df)) % 2 == 0, "F", "M")
        out = samd_fairness_score(df, site_attribute="site", demographic_attributes=["sex"])
        self.assertGreater(out["samd_fairness_score"], 0.75)
        self.assertIn("components", out)
        self.assertIn("weights", out)

    def test_biased_score_is_lower(self) -> None:
        df_bal = _make_balanced_df()
        df_bias = _make_biased_df()
        df_bal["sex"] = "F"
        df_bias["sex"] = "F"
        s_bal = samd_fairness_score(df_bal, demographic_attributes=[])["samd_fairness_score"]
        s_bias = samd_fairness_score(df_bias, demographic_attributes=[])["samd_fairness_score"]
        self.assertGreater(s_bal, s_bias)

    def test_score_is_in_unit_interval(self) -> None:
        df = _make_biased_df()
        df["sex"] = "F"
        out = samd_fairness_score(df, demographic_attributes=["sex"])
        self.assertGreaterEqual(out["samd_fairness_score"], 0.0)
        self.assertLessEqual(out["samd_fairness_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
