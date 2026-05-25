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
