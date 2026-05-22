"""Tests for the BCa bootstrap, permutation test, and MDE primitives.

Strategy: toy distributions with known properties.
- gap = 0 fixture: permutation p-value must be > 0.05 (no detectable gap).
- gap = 0.30 fixture with adequate n: p-value must be < 0.01.
- BCa CI bracket the true gap on a synthetic case with a known shape.
- MDE is monotone in bootstrap_se: tighter SE -> smaller MDE.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fmm_fairness.metrics import _weighted_f1, weighted_f1_gap
from fmm_fairness.statistics import (
    bca_bootstrap_gap_ci,
    cohens_d,
    gap_inference,
    minimum_detectable_effect,
    odds_ratio_binary,
    percentile_bootstrap_gap_ci,
    permutation_test_gap_pvalue,
)

RNG_SEED = 7


def _make_no_gap_df(n: int = 200) -> pd.DataFrame:
    """Two groups, identical Bernoulli distributions => gap ~ 0."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for site in ("A", "B"):
        y_true = rng.integers(0, 2, size=n)
        y_pred = y_true.copy()
        flip = rng.random(n) < 0.10
        y_pred[flip] = 1 - y_pred[flip]
        for yt, yp in zip(y_true, y_pred, strict=True):
            rows.append({"y_true": int(yt), "y_pred": int(yp), "site": site})
    return pd.DataFrame(rows)


def _make_clear_gap_df(n: int = 300) -> pd.DataFrame:
    """Site A: very accurate. Site B: very inaccurate. gap >> 0."""
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    # Site A: 5% error
    y_true_a = rng.integers(0, 2, size=n)
    y_pred_a = y_true_a.copy()
    flip = rng.random(n) < 0.05
    y_pred_a[flip] = 1 - y_pred_a[flip]
    for yt, yp in zip(y_true_a, y_pred_a, strict=True):
        rows.append({"y_true": int(yt), "y_pred": int(yp), "site": "A"})
    # Site B: 45% error
    y_true_b = rng.integers(0, 2, size=n)
    y_pred_b = y_true_b.copy()
    flip_b = rng.random(n) < 0.45
    y_pred_b[flip_b] = 1 - y_pred_b[flip_b]
    for yt, yp in zip(y_true_b, y_pred_b, strict=True):
        rows.append({"y_true": int(yt), "y_pred": int(yp), "site": "B"})
    return pd.DataFrame(rows)


def _f1_fn(K: int):
    def inner(s: pd.DataFrame) -> float:
        return _weighted_f1(s["y_true"].to_numpy(), s["y_pred"].to_numpy(), K)

    return inner


class TestPercentileVsBca(unittest.TestCase):
    def test_both_return_valid_intervals_on_clear_gap(self) -> None:
        df = _make_clear_gap_df()
        fn = _f1_fn(2)
        lo_p, hi_p, se_p, _ = percentile_bootstrap_gap_ci(
            df, "site", fn, n_iters=300
        )
        lo_b, hi_b, se_b, _ = bca_bootstrap_gap_ci(
            df, "site", fn, n_iters=300
        )
        for lo, hi in ((lo_p, hi_p), (lo_b, hi_b)):
            self.assertLess(lo, hi)
            self.assertGreater(lo, 0.05)
            self.assertLess(hi, 0.95)
        self.assertAlmostEqual(se_p, se_b, places=2)


class TestPermutationTest(unittest.TestCase):
    def test_no_gap_returns_large_p(self) -> None:
        df = _make_no_gap_df()
        fn = _f1_fn(2)
        p, _ = permutation_test_gap_pvalue(df, "site", fn, n_iters=300)
        self.assertGreater(p, 0.05)

    def test_clear_gap_returns_small_p(self) -> None:
        df = _make_clear_gap_df()
        fn = _f1_fn(2)
        p, _ = permutation_test_gap_pvalue(df, "site", fn, n_iters=300)
        self.assertLess(p, 0.01)


class TestMinimumDetectableEffect(unittest.TestCase):
    def test_mde_is_monotone_in_se(self) -> None:
        mde_tight = minimum_detectable_effect(0.01, alpha=0.05, power=0.80)
        mde_wide = minimum_detectable_effect(0.10, alpha=0.05, power=0.80)
        self.assertLess(mde_tight, mde_wide)

    def test_mde_nan_on_invalid_se(self) -> None:
        self.assertTrue(np.isnan(minimum_detectable_effect(0.0)))
        self.assertTrue(np.isnan(minimum_detectable_effect(float("nan"))))


class TestGapInferenceBundle(unittest.TestCase):
    def test_bca_default_with_permutation(self) -> None:
        df = _make_clear_gap_df()
        fn = _f1_fn(2)
        inf = gap_inference(
            df,
            "site",
            fn,
            bootstrap_method="bca",
            n_bootstrap_iters=200,
            n_permutation_iters=200,
        )
        self.assertEqual(inf.bootstrap_method, "bca")
        self.assertGreater(inf.gap, 0.1)
        self.assertIsNotNone(inf.permutation_p_value)
        assert inf.permutation_p_value is not None
        self.assertLess(inf.permutation_p_value, 0.05)
        assert inf.minimum_detectable_effect is not None
        self.assertGreater(inf.minimum_detectable_effect, 0.0)

    def test_invalid_bootstrap_method_raises(self) -> None:
        df = _make_clear_gap_df()
        with self.assertRaises(ValueError):
            gap_inference(df, "site", _f1_fn(2), bootstrap_method="bayes")


class TestWeightedF1GapBcaIntegration(unittest.TestCase):
    def test_default_carries_bca_metadata(self) -> None:
        df = _make_clear_gap_df()
        r = weighted_f1_gap(df, "site", bootstrap_iters=200, permutation_iters=200)
        self.assertEqual(r.bootstrap_method, "bca")
        self.assertIsNotNone(r.bootstrap_se)
        self.assertIsNotNone(r.permutation_p_value)
        self.assertIsNotNone(r.minimum_detectable_effect)
        # Serialise: all five new fields must round-trip into the dict.
        d = r.to_dict()
        for k in (
            "bootstrap_method",
            "bootstrap_se",
            "permutation_p_value",
            "permutation_iters",
            "minimum_detectable_effect",
            "alpha",
            "power",
        ):
            self.assertIn(k, d, f"missing field in to_dict(): {k}")

    def test_percentile_fallback_path(self) -> None:
        df = _make_clear_gap_df()
        r = weighted_f1_gap(
            df,
            "site",
            bootstrap_iters=200,
            bootstrap_method="percentile",
        )
        self.assertEqual(r.bootstrap_method, "percentile")


class TestEffectSizes(unittest.TestCase):
    def test_cohens_d_known_case(self) -> None:
        # Two groups separated by 1 SD => d ~ 1.0
        rng = np.random.default_rng(99)
        a = rng.normal(0.0, 1.0, size=200)
        b = rng.normal(1.0, 1.0, size=200)
        d = cohens_d(b, a)  # b - a, positive
        self.assertAlmostEqual(d, 1.0, delta=0.20)

    def test_odds_ratio_known_case(self) -> None:
        # 2x2: group A: 80/100 success; group B: 50/100 success.
        # OR = (80/20) / (50/50) = 4 / 1 = 4.0
        self.assertAlmostEqual(odds_ratio_binary(80, 100, 50, 100), 4.0, places=10)


if __name__ == "__main__":
    unittest.main()
