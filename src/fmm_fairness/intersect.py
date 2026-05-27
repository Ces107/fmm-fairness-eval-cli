"""Intersectional fairness — cross-product groups across protected attributes.

Single-axis fairness (site, sex, age separately) can mask intersectional
disparities (e.g. older women at site HCUV underperform even when the
single-axis site gap and single-axis sex gap each look acceptable). This
module evaluates each requested cross-product as a synthetic attribute and
reports the same gap statistics + small-cell handling on top.

Public API
----------
- ``parse_intersect_spec(spec)``    -> list[list[str]]
- ``add_intersection_columns(df, intersections)`` -> df with new columns
- ``intersectional_weighted_f1_gap(...)`` -> FairnessResult on the synthetic attribute
- ``build_intersectional_breakdown(df, cfg)`` -> evidence-pack-ready dict

Small-cell handling
-------------------
- Cells with ``n < min_group_n`` are excluded with a warning (same convention
  as the single-axis metrics).
- Optional Bayesian shrinkage: cells with ``min_group_n <= n < shrinkage_pivot``
  have their per-group F1 shrunk toward the global F1 using weight
  ``n / (n + shrinkage_kappa)``. This is off by default; reported numbers stay
  empirical unless the caller flips ``shrinkage_kappa`` to a positive value.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fmm_fairness.metrics import (
    MIN_GROUP_N_DEFAULT,
    FairnessResult,
    GroupMetric,
    _macro_f1,
    _weighted_f1,
    detect_num_classes,
)

INTERSECT_SEP = "*"
INTERSECT_LIST_SEP = ","
INTERSECT_COL_PREFIX = "_intersect_"
SHRINKAGE_PIVOT_DEFAULT = 50
SHRINKAGE_KAPPA_DEFAULT = 0  # 0 disables; set >0 to enable


@dataclass
class IntersectionalResult:
    """Output of a single cross-product evaluation (one synthetic attribute)."""

    intersection: list[str]
    synthetic_attribute: str
    metric_name: str
    per_cell: list[GroupMetric] = field(default_factory=list)
    gap: float = 0.0
    excluded_cells: list[str] = field(default_factory=list)
    shrunk_cells: list[str] = field(default_factory=list)
    shrinkage_kappa: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "intersection": self.intersection,
            "synthetic_attribute": self.synthetic_attribute,
            "metric": self.metric_name,
            "gap": float(self.gap),
            "per_cell": [g.to_dict() for g in self.per_cell],
            "excluded_cells": self.excluded_cells,
            "shrunk_cells": self.shrunk_cells,
            "shrinkage_kappa": self.shrinkage_kappa,
        }


def parse_intersect_spec(spec: str | None) -> list[list[str]]:
    """Parse e.g. ``"site*sex,site*age_bucket"`` -> ``[["site","sex"],["site","age_bucket"]]``.

    Empty / None input returns ``[]``. Whitespace is tolerated; duplicate axes
    within a single intersection raise ``ValueError`` (``site*site`` is not
    a meaningful cross-product).
    """
    if not spec or not spec.strip():
        return []
    out: list[list[str]] = []
    for raw in spec.split(INTERSECT_LIST_SEP):
        axes = [a.strip() for a in raw.split(INTERSECT_SEP) if a.strip()]
        if len(axes) < 2:
            raise ValueError(
                f"Intersect spec '{raw}' must list at least two axes separated by '{INTERSECT_SEP}'."
            )
        if len(set(axes)) != len(axes):
            raise ValueError(
                f"Intersect spec '{raw}' contains a duplicate axis. Use distinct attributes."
            )
        out.append(axes)
    return out


def _synthetic_attribute_name(axes: list[str]) -> str:
    return INTERSECT_COL_PREFIX + INTERSECT_SEP.join(axes)


def _canonical_str(value: Any) -> str:
    """Canonicalize an axis value to a string so equivalent numeric forms collapse.

    Without this, a CSV with mixed dtype across rows (``age_bucket=65`` as int
    on one row and ``65.0`` as float on another, e.g. after pandas re-infers a
    column with one NaN) silently produces two distinct cells for the same
    conceptual subgroup (TD-001).

    Convention:
    - NaN / pd.NA → ``"NA"``
    - integer (Python or numpy) → ``str(int(value))``
    - float that is integer-valued → ``str(int(value))``   (65.0 → "65")
    - float with fractional part   → ``repr(value)``       (65.5 → "65.5")
    - everything else              → ``str(value)``
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if v.is_integer():
            return str(int(v))
        return repr(v)
    return str(value)


def add_intersection_columns(
    df: pd.DataFrame, intersections: list[list[str]]
) -> tuple[pd.DataFrame, list[str]]:
    """Add one synthetic column per intersection. Return (new df, synthetic-col names).

    The new column joins the per-axis string representations with ``INTERSECT_SEP``.
    Per-axis values are canonicalised via ``_canonical_str`` so that numerically
    equivalent values (e.g. ``65`` and ``65.0``) collapse to the same cell name.
    Missing axis cells are tolerated (NaN-bearing rows are stamped as ``"NA"``
    on that axis); the caller's small-cell guard then prunes them naturally.
    """
    if not intersections:
        return df, []
    df = df.copy()
    new_cols: list[str] = []
    for axes in intersections:
        missing = [a for a in axes if a not in df.columns]
        if missing:
            raise ValueError(
                f"Intersection axes {missing} not in DataFrame columns; "
                f"declare them with --protected-attrs or fix the CSV."
            )
        syn = _synthetic_attribute_name(axes)
        canonical_axes = pd.DataFrame(
            {a: df[a].map(_canonical_str) for a in axes},
            index=df.index,
        )
        df[syn] = canonical_axes.agg(INTERSECT_SEP.join, axis=1)
        new_cols.append(syn)
    return df, new_cols


def _apply_shrinkage(
    per_cell: list[GroupMetric],
    global_value: float,
    shrinkage_kappa: int,
    shrinkage_pivot: int,
) -> list[str]:
    """Mutate ``per_cell`` in-place applying Bayesian shrinkage to small cells.

    Returns the list of synthetic-attribute names that were shrunk. Shrinkage
    formula is the canonical empirical-Bayes weighted mean:

        v_shrunk = (n * v_empirical + kappa * v_global) / (n + kappa)

    Cells with ``n >= shrinkage_pivot`` are left untouched; cells already
    below ``min_group_n`` should be excluded upstream and never reach here.
    """
    if shrinkage_kappa <= 0:
        return []
    shrunk: list[str] = []
    for cell in per_cell:
        if cell.n >= shrinkage_pivot:
            continue
        if np.isnan(cell.value):
            continue
        new_value = (cell.n * cell.value + shrinkage_kappa * global_value) / (
            cell.n + shrinkage_kappa
        )
        cell.value = float(new_value)
        shrunk.append(cell.group)
    return shrunk


def _gap_from_cells(per_cell: list[GroupMetric]) -> float:
    vals = [c.value for c in per_cell if not np.isnan(c.value)]
    if len(vals) < 2:
        return 0.0
    return float(max(vals) - min(vals))


def _per_cell_f1(
    df: pd.DataFrame,
    synthetic_attr: str,
    num_classes: int,
    min_group_n: int,
    metric: str,
) -> tuple[list[GroupMetric], list[str]]:
    """Compute per-cell {weighted,macro} F1 with small-cell filtering.

    Returns ``(per_cell_metrics, excluded_cell_names)``.
    """
    if metric not in {"weighted", "macro"}:
        raise ValueError(f"metric must be 'weighted' or 'macro', got {metric!r}")
    f1_fn = _weighted_f1 if metric == "weighted" else _macro_f1
    per_cell: list[GroupMetric] = []
    excluded: list[str] = []
    for cell_name, sub in df.groupby(synthetic_attr):
        if len(sub) < min_group_n:
            excluded.append(str(cell_name))
            continue
        value = f1_fn(
            sub["y_true"].to_numpy(),
            sub["y_pred"].to_numpy(),
            num_classes,
        )
        per_cell.append(GroupMetric(group=str(cell_name), n=len(sub), value=value))
    if excluded:
        warnings.warn(
            f"Excluding {len(excluded)} intersection cells of '{synthetic_attr}' with "
            f"n < {min_group_n}: {excluded}",
            stacklevel=2,
        )
    return per_cell, excluded


def intersectional_f1_gap(
    df: pd.DataFrame,
    axes: list[str],
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    metric: str = "weighted",
    shrinkage_kappa: int = SHRINKAGE_KAPPA_DEFAULT,
    shrinkage_pivot: int = SHRINKAGE_PIVOT_DEFAULT,
) -> IntersectionalResult:
    """Cross-product F1 gap.

    ``axes`` is the list of protected-attribute column names whose cartesian
    product defines the cells. ``metric`` chooses between weighted F1 (default,
    matches the headline single-axis metric) and macro F1 (more sensitive to
    minority classes).

    Bayesian shrinkage is opt-in: pass ``shrinkage_kappa > 0`` to pull
    sub-pivot cells toward the global F1. The shrunk cell names are recorded
    in the result for audit transparency.
    """
    K = detect_num_classes(df, num_classes)
    df_with, syn_cols = add_intersection_columns(df, [axes])
    syn = syn_cols[0]
    per_cell, excluded = _per_cell_f1(df_with, syn, K, min_group_n, metric)
    # Global F1 over the union of retained cells; this is the shrinkage anchor.
    if per_cell:
        retained_idx = df_with[df_with[syn].isin([c.group for c in per_cell])].index
        if metric == "weighted":
            global_value = _weighted_f1(
                df_with.loc[retained_idx, "y_true"].to_numpy(),
                df_with.loc[retained_idx, "y_pred"].to_numpy(),
                K,
            )
        else:
            global_value = _macro_f1(
                df_with.loc[retained_idx, "y_true"].to_numpy(),
                df_with.loc[retained_idx, "y_pred"].to_numpy(),
                K,
            )
    else:
        global_value = float("nan")
    shrunk = (
        _apply_shrinkage(per_cell, global_value, shrinkage_kappa, shrinkage_pivot)
        if not np.isnan(global_value)
        else []
    )
    gap = _gap_from_cells(per_cell)
    metric_name = f"intersectional_{metric}_f1_gap"
    return IntersectionalResult(
        intersection=list(axes),
        synthetic_attribute=syn,
        metric_name=metric_name,
        per_cell=per_cell,
        gap=gap,
        excluded_cells=excluded,
        shrunk_cells=shrunk,
        shrinkage_kappa=shrinkage_kappa,
    )


def intersectional_to_fairness_result(ir: IntersectionalResult) -> FairnessResult:
    """Adapter: surface an intersectional cross-product through the standard
    ``FairnessResult`` shape, so existing evidence-pack machinery can render
    it without a special case.
    """
    return FairnessResult(
        metric_name=ir.metric_name,
        attribute=ir.synthetic_attribute,
        per_group=list(ir.per_cell),
        gap=ir.gap,
        excluded_groups=ir.excluded_cells,
    )


def build_intersectional_breakdown(
    df: pd.DataFrame,
    intersections: list[list[str]],
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    shrinkage_kappa: int = SHRINKAGE_KAPPA_DEFAULT,
    shrinkage_pivot: int = SHRINKAGE_PIVOT_DEFAULT,
) -> dict[str, Any]:
    """Top-level orchestrator. Returns the evidence-pack ``intersectional_breakdown`` dict."""
    K = detect_num_classes(df, num_classes)
    if not intersections:
        return {
            "intersections_declared": [],
            "min_group_n": min_group_n,
            "shrinkage_kappa": shrinkage_kappa,
            "shrinkage_pivot": shrinkage_pivot,
            "results": [],
        }
    results: list[dict[str, Any]] = []
    for axes in intersections:
        weighted = intersectional_f1_gap(
            df,
            axes,
            num_classes=K,
            min_group_n=min_group_n,
            metric="weighted",
            shrinkage_kappa=shrinkage_kappa,
            shrinkage_pivot=shrinkage_pivot,
        )
        macro = intersectional_f1_gap(
            df,
            axes,
            num_classes=K,
            min_group_n=min_group_n,
            metric="macro",
            shrinkage_kappa=shrinkage_kappa,
            shrinkage_pivot=shrinkage_pivot,
        )
        results.append(
            {
                "axes": list(axes),
                "synthetic_attribute": weighted.synthetic_attribute,
                "weighted_f1_gap": weighted.to_dict(),
                "macro_f1_gap": macro.to_dict(),
            }
        )
    return {
        "intersections_declared": [list(a) for a in intersections],
        "min_group_n": min_group_n,
        "shrinkage_kappa": shrinkage_kappa,
        "shrinkage_pivot": shrinkage_pivot,
        "results": results,
    }
