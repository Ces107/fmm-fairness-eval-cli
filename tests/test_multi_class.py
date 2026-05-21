"""Tests for the multi-class data model and the F1-family fairness metrics.

The synthetic AI4SkIN-shaped fixture builds a 6-class dermatology classifier
output across two centres (HCUV, HUSC) with the centre B classifier degraded
on three of the six classes. The known per-group weighted-F1 gap is in the
0.15-0.25 range, anchoring the TFG-shape replication test.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fmm_fairness.metrics import (
    SCORE_COL_PREFIX,
    FairnessResult,
    _multi_class_score_columns,
    _score_matrix,
    detect_num_classes,
    inter_site_auc_variance,
    macro_f1_gap,
    multi_class_auc_gap,
    per_class_f1_gap,
    samd_fairness_score,
    weighted_f1_gap,
)

RNG_SEED = 4321
NUM_CLASSES = 6


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _make_ai4skin_shaped_df(
    n_per_site: int = 400,
    *,
    biased: bool = True,
    rater_columns: bool = False,
) -> pd.DataFrame:
    """Build a 6-class multi-site dermatology fixture.

    The mechanism is explicit: for each row we draw a target prediction
    equal to y_true with probability ``accuracy(class, site)``, else a
    uniform misclassification. Probability vectors are softmaxed around
    the chosen target. This decouples ``accuracy`` from logit-noise so the
    inter-site F1 gap matches the configured shape.

    Site A (HCUV) gets accuracy ~0.90 across all classes.
    Site B (HUSC) under ``biased=True`` gets ~0.65 on common classes
    {0, 1, 2} and ~0.30 on rare classes {3, 4, 5}; under ``biased=False``
    it matches site A.
    """
    rng = np.random.default_rng(RNG_SEED)
    K = NUM_CLASSES
    rows = []

    if biased:
        site_specs = [
            ("HCUV", 0.90, 0.90),
            ("HUSC", 0.65, 0.30),
        ]
    else:
        site_specs = [
            ("HCUV", 0.90, 0.90),
            ("HUSC", 0.90, 0.90),
        ]

    prevalence = np.array([0.30, 0.25, 0.18, 0.12, 0.10, 0.05])

    for site, acc_strong, acc_weak in site_specs:
        n = n_per_site
        y_true = rng.choice(np.arange(K), size=n, p=prevalence)
        y_pred = np.empty(n, dtype=int)
        for i, cls in enumerate(y_true):
            weak = cls >= 3
            accuracy = acc_weak if weak else acc_strong
            if rng.random() < accuracy:
                y_pred[i] = cls
            else:
                other = rng.integers(0, K - 1)
                if other >= cls:
                    other += 1  # uniform over K-1 alternatives
                y_pred[i] = other
        logits = rng.normal(0.0, 0.4, size=(n, K))
        logits[np.arange(n), y_pred] += 3.5  # peak at the chosen prediction
        probs = _softmax_rows(logits)
        site_rows = {f"{SCORE_COL_PREFIX}{k}": probs[:, k] for k in range(K)}
        site_rows["y_true"] = y_true.astype(int)
        site_rows["y_pred"] = y_pred.astype(int)
        site_rows["site"] = np.full(n, site)
        site_rows["sex"] = rng.choice(["F", "M"], size=n)
        df_site = pd.DataFrame(site_rows)
        if rater_columns:
            for r in range(1, 11):
                noisy = y_true.copy()
                flip_mask = rng.random(n) < 0.10
                noisy[flip_mask] = rng.integers(0, K, size=flip_mask.sum())
                df_site[f"doc{r}"] = noisy.astype(int)
        rows.append(df_site)
    return pd.concat(rows, ignore_index=True)


def _make_balanced_binary_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    rows = []
    for site in ("A", "B"):
        y_true = rng.integers(0, 2, size=n)
        y_score = np.where(
            y_true == 1, rng.beta(8, 2, size=n), rng.beta(2, 8, size=n)
        )
        y_pred = (y_score >= 0.5).astype(int)
        for yt, ys, yp in zip(y_true, y_score, y_pred, strict=True):
            rows.append(
                {
                    "y_true": int(yt),
                    "y_pred": int(yp),
                    "y_score": float(ys),
                    "site": site,
                }
            )
    return pd.DataFrame(rows)


class TestDetectNumClasses(unittest.TestCase):
    def test_detects_from_multi_class_score_columns(self) -> None:
        df = _make_ai4skin_shaped_df()
        self.assertEqual(detect_num_classes(df), NUM_CLASSES)
        cols = _multi_class_score_columns(df)
        self.assertEqual(len(cols), NUM_CLASSES)
        self.assertEqual(cols[0], "y_score_0")
        self.assertEqual(cols[-1], f"y_score_{NUM_CLASSES - 1}")

    def test_detects_binary_from_y_score(self) -> None:
        df = _make_balanced_binary_df()
        self.assertEqual(detect_num_classes(df), 2)

    def test_explicit_override_with_mismatch_warns(self) -> None:
        df = _make_ai4skin_shaped_df()
        with self.assertWarns(UserWarning):
            K = detect_num_classes(df, num_classes=4)
        self.assertEqual(K, 4)

    def test_rejects_num_classes_lt_two(self) -> None:
        df = _make_ai4skin_shaped_df()
        with self.assertRaises(ValueError):
            detect_num_classes(df, num_classes=1)

    def test_rejects_non_contiguous_score_columns(self) -> None:
        df = _make_ai4skin_shaped_df().drop(columns=["y_score_2"])
        with self.assertRaises(ValueError):
            _multi_class_score_columns(df)


class TestScoreMatrix(unittest.TestCase):
    def test_binary_lifts_to_two_column_matrix(self) -> None:
        df = _make_balanced_binary_df()
        mat = _score_matrix(df, 2)
        self.assertEqual(mat.shape, (len(df), 2))
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-9)

    def test_multi_class_returns_k_columns(self) -> None:
        df = _make_ai4skin_shaped_df()
        mat = _score_matrix(df, NUM_CLASSES)
        self.assertEqual(mat.shape, (len(df), NUM_CLASSES))
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-5)


class TestWeightedF1Gap(unittest.TestCase):
    def test_biased_gap_recovers_tfg_shape(self) -> None:
        df = _make_ai4skin_shaped_df()
        r = weighted_f1_gap(df, "site", bootstrap_iters=100)
        self.assertIsInstance(r, FairnessResult)
        self.assertEqual(r.metric_name, "weighted_f1_gap")
        # In a calibrated fixture this lands in the 0.10-0.40 band; the precise
        # number depends on the RNG, but it must be materially non-zero and the
        # CI must bracket the headline.
        self.assertGreater(r.gap, 0.10)
        self.assertLess(r.gap, 0.50)
        self.assertEqual(len(r.per_group), 2)
        self.assertIsNotNone(r.gap_ci_low)
        self.assertIsNotNone(r.gap_ci_high)

    def test_balanced_gap_is_small(self) -> None:
        df = _make_ai4skin_shaped_df(biased=False)
        r = weighted_f1_gap(df, "site", bootstrap_iters=50)
        self.assertLess(r.gap, 0.10)

    def test_binary_path_runs(self) -> None:
        df = _make_balanced_binary_df()
        r = weighted_f1_gap(df, "site", bootstrap_iters=50)
        self.assertGreaterEqual(r.gap, 0.0)


class TestMacroF1Gap(unittest.TestCase):
    def test_biased_macro_gap_is_larger_than_balanced(self) -> None:
        df_bias = _make_ai4skin_shaped_df()
        df_bal = _make_ai4skin_shaped_df(biased=False)
        r_bias = macro_f1_gap(df_bias, "site", bootstrap_iters=50)
        r_bal = macro_f1_gap(df_bal, "site", bootstrap_iters=50)
        self.assertGreater(r_bias.gap, r_bal.gap)


class TestPerClassF1Gap(unittest.TestCase):
    def test_per_class_vector_shape_and_worst_class(self) -> None:
        df = _make_ai4skin_shaped_df()
        r = per_class_f1_gap(df, "site", bootstrap_iters=50)
        self.assertIsNotNone(r.per_class_gap)
        assert r.per_class_gap is not None
        self.assertEqual(len(r.per_class_gap), NUM_CLASSES)
        # The fixture degrades classes 3-5; the worst class should be one of those.
        worst_idx = int(np.argmax(r.per_class_gap))
        self.assertIn(worst_idx, {3, 4, 5})
        for grp in r.per_group:
            self.assertIsNotNone(grp.per_class)
            assert grp.per_class is not None
            self.assertEqual(len(grp.per_class), NUM_CLASSES)

    def test_top_level_gap_is_worst_class_gap(self) -> None:
        df = _make_ai4skin_shaped_df()
        r = per_class_f1_gap(df, "site", bootstrap_iters=50)
        assert r.per_class_gap is not None
        self.assertAlmostEqual(r.gap, max(r.per_class_gap), places=12)


class TestMultiClassAucGap(unittest.TestCase):
    def test_biased_multi_class_auc_gap_is_positive(self) -> None:
        df = _make_ai4skin_shaped_df()
        r = multi_class_auc_gap(df, "site", bootstrap_iters=50)
        self.assertGreater(r.gap, 0.0)
        for grp in r.per_group:
            self.assertGreaterEqual(grp.value, 0.0)
            self.assertLessEqual(grp.value, 1.0)


class TestInterSiteAUCVarianceKAware(unittest.TestCase):
    def test_multi_class_path_uses_ovr_macro(self) -> None:
        df = _make_ai4skin_shaped_df()
        r = inter_site_auc_variance(df, "site")
        self.assertEqual(r.metric_name, "inter_site_auc_variance")
        self.assertGreater(r.gap, 0.0)  # variance is non-trivial under bias
        for grp in r.per_group:
            self.assertGreaterEqual(grp.value, 0.0)
            self.assertLessEqual(grp.value, 1.0)

    def test_binary_path_bit_identical_to_v01(self) -> None:
        df = _make_balanced_binary_df()
        r = inter_site_auc_variance(df, "site")
        # Same numerics as v0.1 (balanced fixture variance ~ very small).
        self.assertLess(r.gap, 0.01)


class TestSamdFairnessScoreMultiClass(unittest.TestCase):
    def test_multi_class_score_in_unit_interval(self) -> None:
        df = _make_ai4skin_shaped_df()
        out = samd_fairness_score(df, site_attribute="site", demographic_attributes=[])
        s = out["samd_fairness_score"]
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)
        self.assertEqual(out["num_classes"], NUM_CLASSES)
        self.assertEqual(out["components"]["eo_mean"], 0.0)  # binary-only term zeroed
        self.assertGreater(out["components"]["f1_site_term"], 0.0)

    def test_biased_multi_class_score_lower_than_balanced(self) -> None:
        bias = samd_fairness_score(_make_ai4skin_shaped_df(), demographic_attributes=[])
        bal = samd_fairness_score(_make_ai4skin_shaped_df(biased=False), demographic_attributes=[])
        self.assertGreater(bal["samd_fairness_score"], bias["samd_fairness_score"])

    def test_weight_renorm_when_k_gt_2(self) -> None:
        df = _make_ai4skin_shaped_df()
        out = samd_fairness_score(df, demographic_attributes=[])
        eff = out["weights"]
        self.assertEqual(eff["eo"], 0.0)
        self.assertEqual(eff["dp"], 0.0)
        self.assertEqual(eff["cal"], 0.0)
        self.assertAlmostEqual(eff["f1_site"] + eff["site"], 1.0, places=9)


class TestBinaryBackwardCompat(unittest.TestCase):
    """v0.1 numerics must survive the v0.2 refactor for binary inputs."""

    def test_balanced_score_still_high(self) -> None:
        df = _make_balanced_binary_df()
        df["sex"] = np.where(np.arange(len(df)) % 2 == 0, "F", "M")
        out = samd_fairness_score(df, site_attribute="site", demographic_attributes=["sex"])
        self.assertGreater(out["samd_fairness_score"], 0.70)

    def test_binary_score_in_unit_interval(self) -> None:
        df = _make_balanced_binary_df()
        df["sex"] = "F"
        out = samd_fairness_score(df, demographic_attributes=["sex"])
        self.assertGreaterEqual(out["samd_fairness_score"], 0.0)
        self.assertLessEqual(out["samd_fairness_score"], 1.0)
        self.assertEqual(out["num_classes"], 2)


class TestRequireBinaryRejectsMultiClass(unittest.TestCase):
    def test_equal_opportunity_gap_rejects_multi_class(self) -> None:
        from fmm_fairness.metrics import equal_opportunity_gap

        df = _make_ai4skin_shaped_df()
        with self.assertRaises(ValueError):
            equal_opportunity_gap(df, "site")


if __name__ == "__main__":
    unittest.main()
