"""Tests for fmm_fairness.intersect."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmm_fairness.intersect import (
    INTERSECT_COL_PREFIX,
    add_intersection_columns,
    build_intersectional_breakdown,
    intersectional_f1_gap,
    intersectional_to_fairness_result,
    parse_intersect_spec,
)


def _make_df(seed: int = 20260525, n: int = 240, *, disparity: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    site = rng.choice(["A", "B"], size=n)
    sex = rng.choice(["M", "F"], size=n)
    y_true = rng.integers(0, 3, size=n)
    y_pred = y_true.copy()
    if disparity:
        flip = (site == "A") & (sex == "F")
        y_pred[flip] = (y_pred[flip] + 1) % 3
    y_score = np.full((n, 3), 0.2)
    y_score[np.arange(n), y_pred] = 0.6
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score_0": y_score[:, 0],
            "y_score_1": y_score[:, 1],
            "y_score_2": y_score[:, 2],
            "site": site,
            "sex": sex,
        }
    )
    return df


class TestParseIntersectSpec:
    def test_none_and_empty_return_empty_list(self) -> None:
        assert parse_intersect_spec(None) == []
        assert parse_intersect_spec("") == []
        assert parse_intersect_spec("   ") == []

    def test_single_intersection(self) -> None:
        assert parse_intersect_spec("site*sex") == [["site", "sex"]]

    def test_multiple_intersections_and_whitespace(self) -> None:
        spec = "site * sex , site*age_bucket"
        assert parse_intersect_spec(spec) == [
            ["site", "sex"],
            ["site", "age_bucket"],
        ]

    def test_single_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two axes"):
            parse_intersect_spec("site")

    def test_duplicate_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate axis"):
            parse_intersect_spec("site*site")


class TestAddIntersectionColumns:
    def test_adds_synthetic_column(self) -> None:
        df = _make_df()
        new_df, new_cols = add_intersection_columns(df, [["site", "sex"]])
        assert len(new_cols) == 1
        col = new_cols[0]
        assert col.startswith(INTERSECT_COL_PREFIX)
        # original df untouched
        assert col not in df.columns
        # synthetic value is the join of axis strings
        for _, row in new_df.iterrows():
            assert row[col] == f"{row['site']}*{row['sex']}"

    def test_missing_axis_raises(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="not in DataFrame"):
            add_intersection_columns(df, [["site", "missing_axis"]])


class TestIntersectionalF1Gap:
    def test_detects_intersectional_disparity(self) -> None:
        df = _make_df(disparity=True)
        result = intersectional_f1_gap(df, ["site", "sex"], num_classes=3)
        # 4 cells (A*F, A*M, B*F, B*M); A*F is wrong on everything
        cells = {c.group: c for c in result.per_cell}
        assert set(cells) == {"A*F", "A*M", "B*F", "B*M"}
        assert cells["A*F"].value < cells["A*M"].value
        # synthetic attribute name carries the intersection
        assert "site" in result.synthetic_attribute
        assert "sex" in result.synthetic_attribute
        assert result.gap > 0.5

    def test_no_disparity_yields_small_gap(self) -> None:
        df = _make_df(disparity=False)
        result = intersectional_f1_gap(df, ["site", "sex"], num_classes=3)
        # perfect predictions everywhere
        for c in result.per_cell:
            assert c.value == pytest.approx(1.0, abs=1e-12)
        assert result.gap == pytest.approx(0.0, abs=1e-12)

    def test_small_cell_excluded(self) -> None:
        df = _make_df()
        # add a tiny 3rd site
        extra = df.iloc[:5].copy()
        extra["site"] = "C"
        df_with_small = pd.concat([df, extra], ignore_index=True)
        result = intersectional_f1_gap(
            df_with_small, ["site", "sex"], num_classes=3, min_group_n=20
        )
        # C*F and C*M cells are below min_group_n and excluded
        excluded = set(result.excluded_cells)
        assert any(c.startswith("C*") for c in excluded)
        # retained cells are still 4 (A*F, A*M, B*F, B*M)
        assert len(result.per_cell) == 4

    def test_shrinkage_pulls_small_cells_toward_global(self) -> None:
        df = _make_df(disparity=True)
        # Without shrinkage
        no_shrink = intersectional_f1_gap(
            df, ["site", "sex"], num_classes=3, shrinkage_kappa=0
        )
        # With aggressive shrinkage (kappa very large) every cell collapses to global
        heavy_shrink = intersectional_f1_gap(
            df,
            ["site", "sex"],
            num_classes=3,
            shrinkage_kappa=10_000,
            shrinkage_pivot=10_000,  # force all cells to be shrunk
        )
        # gap shrinks with shrinkage
        assert heavy_shrink.gap < no_shrink.gap
        # all cells are marked as shrunk
        assert len(heavy_shrink.shrunk_cells) == len(heavy_shrink.per_cell)

    def test_macro_metric(self) -> None:
        df = _make_df(disparity=True)
        result = intersectional_f1_gap(
            df, ["site", "sex"], num_classes=3, metric="macro"
        )
        assert result.metric_name == "intersectional_macro_f1_gap"

    def test_unknown_metric_raises(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="metric must"):
            intersectional_f1_gap(df, ["site", "sex"], num_classes=3, metric="other")


class TestBuildIntersectionalBreakdown:
    def test_empty_returns_empty_results(self) -> None:
        df = _make_df()
        block = build_intersectional_breakdown(df, [], num_classes=3)
        assert block["intersections_declared"] == []
        assert block["results"] == []

    def test_multi_intersection(self) -> None:
        df = _make_df()
        df["age_bucket"] = np.where(np.arange(len(df)) % 2 == 0, "lt60", "gte60")
        block = build_intersectional_breakdown(
            df,
            [["site", "sex"], ["site", "age_bucket"]],
            num_classes=3,
        )
        assert len(block["results"]) == 2
        for res in block["results"]:
            assert "weighted_f1_gap" in res and "macro_f1_gap" in res
            assert res["weighted_f1_gap"]["metric"] == "intersectional_weighted_f1_gap"
            assert res["macro_f1_gap"]["metric"] == "intersectional_macro_f1_gap"


class TestIntersectionalToFairnessResult:
    def test_adapter_carries_gap_and_cells(self) -> None:
        df = _make_df(disparity=True)
        ir = intersectional_f1_gap(df, ["site", "sex"], num_classes=3)
        fr = intersectional_to_fairness_result(ir)
        assert fr.metric_name == ir.metric_name
        assert fr.gap == ir.gap
        assert len(fr.per_group) == len(ir.per_cell)


class TestIntersectionalInference:
    """TD-002: BCa CI + permutation p-value + MDE on the intersectional gap."""

    def test_inference_disabled_by_default(self) -> None:
        df = _make_df(disparity=True)
        result = intersectional_f1_gap(df, ["site", "sex"], num_classes=3)
        assert result.inference is None
        assert "inference" not in result.to_dict()

    def test_inference_attached_when_bootstrap_iters_gt_zero(self) -> None:
        df = _make_df(disparity=True, n=300)
        result = intersectional_f1_gap(
            df,
            ["site", "sex"],
            num_classes=3,
            bootstrap_iters=200,
            permutation_iters=200,
            seed=7,
        )
        assert result.inference is not None
        inf = result.inference
        assert inf.bootstrap_method in {"bca", "percentile"}
        assert inf.bootstrap_iters == 200
        assert inf.permutation_iters == 200
        assert inf.n_retained_cells >= 2
        assert 0.0 <= inf.permutation_p_value <= 1.0  # type: ignore[operator]
        # MDE can be NaN in the degenerate case where bootstrap SE is 0
        # (e.g. a planted disparity producing identical bootstrap replicates).
        # In the non-degenerate case it must be a positive float.
        if inf.bootstrap_se is not None and inf.bootstrap_se > 0.0:
            assert inf.minimum_detectable_effect is not None
            assert inf.minimum_detectable_effect > 0.0
        as_dict = result.to_dict()["inference"]
        for key in (
            "bootstrap_method",
            "ci_low",
            "ci_high",
            "bootstrap_se",
            "permutation_p_value",
            "minimum_detectable_effect",
            "n_retained_cells",
        ):
            assert key in as_dict

    def test_inference_detects_real_disparity(self) -> None:
        """With a planted A*F disparity, the permutation p should drop sharply.

        With disparity=True the per-cell F1 is depressed only for A*F (one
        of four cells), so the gap is structural, not noise. The two-sided
        permutation p-value on label shuffles should land far below the
        no-disparity baseline.
        """
        df_dis = _make_df(disparity=True, n=400, seed=11)
        df_null = _make_df(disparity=False, n=400, seed=11)
        r_dis = intersectional_f1_gap(
            df_dis,
            ["site", "sex"],
            num_classes=3,
            bootstrap_iters=200,
            permutation_iters=500,
            seed=11,
        )
        r_null = intersectional_f1_gap(
            df_null,
            ["site", "sex"],
            num_classes=3,
            bootstrap_iters=200,
            permutation_iters=500,
            seed=11,
        )
        assert r_dis.inference is not None
        assert r_null.inference is not None
        # Planted gap should look more extreme on the permutation distribution
        # than the no-disparity baseline. We assert the inequality rather than
        # an absolute threshold so the test is robust to minor seed shifts.
        assert (
            r_dis.inference.permutation_p_value  # type: ignore[operator]
            < r_null.inference.permutation_p_value  # type: ignore[operator]
        )
        # And the gap itself should be larger under planted disparity.
        assert r_dis.gap > r_null.gap

    def test_inference_propagates_through_build_intersectional_breakdown(self) -> None:
        df = _make_df(disparity=True, n=300)
        block = build_intersectional_breakdown(
            df,
            [["site", "sex"]],
            num_classes=3,
            bootstrap_iters=200,
            permutation_iters=0,
            seed=7,
        )
        assert block["bootstrap_iters"] == 200
        assert block["permutation_iters"] == 0
        weighted_inf = block["results"][0]["weighted_f1_gap"].get("inference")
        macro_inf = block["results"][0]["macro_f1_gap"].get("inference")
        assert weighted_inf is not None
        assert macro_inf is not None
        assert weighted_inf["bootstrap_method"] in {"bca", "percentile"}
        assert weighted_inf["permutation_p_value"] is None  # we passed 0
