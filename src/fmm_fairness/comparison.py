"""Foundation-model comparison mode.

The underlying TFG compared six foundation-model embeddings (UNI, CONCH,
PLIP, TransPath, GigaPath, TITAN) plus a CONCH+UNI concatenation as the
backbone for a six-class dermatopathology classifier. The differentiator
of this CLI vs FairLearn / AIF360 is that the joint accuracy + fairness
question is the practitioner's actual decision: *"which foundation model
should I deploy across my hospital network?"*

This module implements the answer: a Pareto frontier over
(overall weighted F1, inter-site weighted F1 gap) per candidate model.
A model is on the frontier when no other candidate is both at least as
accurate overall and at least as fair across sites, with strict
improvement on at least one of the two axes.

The output is regulator-readable: the JSON evidence pack documents which
foundation-model candidates form the frontier, which are dominated and
should not be deployed, and a single AI-Act-Art.-9 recommendation that
picks the frontier model with the lowest residual risk at a stated
fairness floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fmm_fairness.metrics import (
    _macro_f1,
    _weighted_f1,
    detect_num_classes,
    macro_f1_gap,
    samd_fairness_score,
    weighted_f1_gap,
)


@dataclass
class ModelEvaluation:
    """Summary of one foundation-model candidate."""

    label: str
    n_samples: int
    num_classes: int
    overall_weighted_f1: float
    overall_macro_f1: float
    weighted_f1_gap_site: float
    macro_f1_gap_site: float
    samd_fairness_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_samples": int(self.n_samples),
            "num_classes": int(self.num_classes),
            "overall_weighted_f1": float(self.overall_weighted_f1),
            "overall_macro_f1": float(self.overall_macro_f1),
            "weighted_f1_gap_site": float(self.weighted_f1_gap_site),
            "macro_f1_gap_site": float(self.macro_f1_gap_site),
            "samd_fairness_score": float(self.samd_fairness_score),
        }


@dataclass
class ComparisonResult:
    """Cross-model comparison artefact."""

    models: list[ModelEvaluation] = field(default_factory=list)
    pareto_frontier_labels: list[str] = field(default_factory=list)
    pareto_dominated_labels: list[str] = field(default_factory=list)
    recommended_label: str | None = None
    recommendation_rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [m.to_dict() for m in self.models],
            "pareto_frontier_labels": list(self.pareto_frontier_labels),
            "pareto_dominated_labels": list(self.pareto_dominated_labels),
            "recommended_label": self.recommended_label,
            "recommendation_rationale": self.recommendation_rationale,
        }


def _pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Return indices of non-dominated (perf, gap) pairs.

    Convention: higher ``perf`` is better, lower ``gap`` is better.
    Point ``j`` dominates point ``i`` iff ``perf_j >= perf_i`` and
    ``gap_j <= gap_i`` with strict inequality on at least one axis.
    """
    n = len(points)
    on_frontier: list[int] = []
    for i in range(n):
        perf_i, gap_i = points[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            perf_j, gap_j = points[j]
            if perf_j >= perf_i and gap_j <= gap_i and (perf_j > perf_i or gap_j < gap_i):
                dominated = True
                break
        if not dominated:
            on_frontier.append(i)
    return on_frontier


def _evaluate_one(
    df: pd.DataFrame,
    label: str,
    *,
    site_attribute: str,
    demographic_attributes: list[str],
    num_classes: int | None,
) -> ModelEvaluation:
    """Compute all per-model summary numbers needed for the comparison."""
    K = detect_num_classes(df, num_classes)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    overall_w = _weighted_f1(y_true, y_pred, K)
    overall_m = _macro_f1(y_true, y_pred, K)
    w_gap = (
        weighted_f1_gap(df, site_attribute, num_classes=K, bootstrap_iters=50).gap
        if site_attribute in df.columns
        else 0.0
    )
    m_gap = (
        macro_f1_gap(df, site_attribute, num_classes=K, bootstrap_iters=50).gap
        if site_attribute in df.columns
        else 0.0
    )
    composite = samd_fairness_score(
        df,
        site_attribute=site_attribute,
        demographic_attributes=demographic_attributes,
        num_classes=K,
    )
    return ModelEvaluation(
        label=label,
        n_samples=len(df),
        num_classes=K,
        overall_weighted_f1=float(overall_w) if not np.isnan(overall_w) else 0.0,
        overall_macro_f1=float(overall_m) if not np.isnan(overall_m) else 0.0,
        weighted_f1_gap_site=float(w_gap),
        macro_f1_gap_site=float(m_gap),
        samd_fairness_score=float(composite["samd_fairness_score"]),
    )


def _recommend_from_frontier(
    models: list[ModelEvaluation],
    frontier_indices: list[int],
    fairness_floor: float,
) -> tuple[str | None, str]:
    """Pick the frontier model with the lowest residual-risk profile.

    Heuristic: among frontier models whose ``weighted_f1_gap_site`` is at
    or below ``fairness_floor``, pick the one with the highest overall
    weighted F1. If none meet the floor, fall back to the frontier model
    with the smallest gap (the "fairest" candidate); flag this in the
    rationale so the operator sees the trade-off explicitly.
    """
    if not frontier_indices:
        return None, "No frontier models could be identified."
    candidates = [models[i] for i in frontier_indices]
    eligible = [m for m in candidates if m.weighted_f1_gap_site <= fairness_floor]
    if eligible:
        winner = max(eligible, key=lambda m: m.overall_weighted_f1)
        return winner.label, (
            f"Frontier model with the highest overall weighted F1 "
            f"({winner.overall_weighted_f1:.4f}) among candidates whose "
            f"inter-site weighted F1 gap is at or below the configured "
            f"fairness floor of {fairness_floor:.2f}. EU AI Act Art. 9 "
            f"residual-risk recommendation."
        )
    winner = min(candidates, key=lambda m: m.weighted_f1_gap_site)
    return winner.label, (
        f"No frontier model meets the configured fairness floor of "
        f"{fairness_floor:.2f}; falling back to the fairest frontier "
        f"candidate (gap = {winner.weighted_f1_gap_site:.4f}). Operator "
        f"review required before deployment; this case typically warrants "
        f"either a tighter site-specific calibration step or a model "
        f"change before the dossier can be signed off."
    )


def compare_models(
    dfs: list[pd.DataFrame],
    labels: list[str],
    *,
    site_attribute: str = "site",
    demographic_attributes: list[str] | None = None,
    num_classes: int | None = None,
    fairness_floor: float = 0.10,
) -> ComparisonResult:
    """Evaluate each (df, label) pair, compute Pareto frontier, recommend a model.

    ``dfs`` and ``labels`` must have the same length. Each DataFrame must
    share the same protected-attribute declarations and a comparable
    ``y_true`` distribution; this is the operator's responsibility (the
    tool does not enforce identical test sets, only checks they are
    parseable).

    ``fairness_floor`` is the maximum acceptable weighted-F1 inter-site
    gap for the AI-Act-Art.-9 recommendation; default 0.10 is a
    conservative starting point and should be tuned per clinical context.
    """
    if len(dfs) != len(labels):
        raise ValueError(
            f"compare_models: got {len(dfs)} DataFrames but {len(labels)} labels."
        )
    if len(dfs) < 2:
        raise ValueError("compare_models needs at least 2 candidates.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate labels in compare_models input: {labels}.")
    if demographic_attributes is None:
        demographic_attributes = []

    models = [
        _evaluate_one(
            df,
            label,
            site_attribute=site_attribute,
            demographic_attributes=demographic_attributes,
            num_classes=num_classes,
        )
        for df, label in zip(dfs, labels, strict=True)
    ]
    points = [(m.overall_weighted_f1, m.weighted_f1_gap_site) for m in models]
    frontier = _pareto_frontier(points)
    frontier_labels = [models[i].label for i in frontier]
    dominated_labels = [m.label for i, m in enumerate(models) if i not in frontier]
    recommended, rationale = _recommend_from_frontier(
        models, frontier, fairness_floor
    )
    return ComparisonResult(
        models=models,
        pareto_frontier_labels=frontier_labels,
        pareto_dominated_labels=dominated_labels,
        recommended_label=recommended,
        recommendation_rationale=rationale,
    )


def render_comparison_markdown(result: ComparisonResult) -> str:
    """Regulator-readable Markdown rendering of a ComparisonResult."""
    lines: list[str] = []
    lines.append("# Foundation-model comparison report")
    lines.append("")
    lines.append(f"- **Candidates evaluated**: {len(result.models)}")
    lines.append(f"- **Pareto frontier**: {', '.join(result.pareto_frontier_labels) or '(empty)'}")
    if result.pareto_dominated_labels:
        lines.append(
            f"- **Dominated (do not deploy as-is)**: {', '.join(result.pareto_dominated_labels)}"
        )
    if result.recommended_label is not None:
        lines.append(f"- **Recommended (Art. 9 residual-risk)**: `{result.recommended_label}`")
        lines.append(f"  - **Rationale**: {result.recommendation_rationale}")
    lines.append("")
    lines.append("## Per-model summary")
    lines.append("")
    lines.append(
        "| Model | n | K | Overall weighted F1 | Overall macro F1 | "
        "Inter-site weighted F1 gap | Inter-site macro F1 gap | "
        "SaMD fairness score | On frontier? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in result.models:
        on_frontier = "yes" if m.label in result.pareto_frontier_labels else "no"
        lines.append(
            f"| `{m.label}` | {m.n_samples} | {m.num_classes} | "
            f"{m.overall_weighted_f1:.4f} | {m.overall_macro_f1:.4f} | "
            f"{m.weighted_f1_gap_site:.4f} | {m.macro_f1_gap_site:.4f} | "
            f"{m.samd_fairness_score:.4f} | {on_frontier} |"
        )
    lines.append("")
    lines.append("## How to read this")
    lines.append(
        "The Pareto frontier collects candidates that are not dominated on "
        "the joint (overall weighted F1, inter-site weighted F1 gap) "
        "axes. A candidate is dominated when another candidate is both at "
        "least as accurate and at least as fair, with strict improvement "
        "on at least one axis. Dominated candidates should not be "
        "deployed as-is; a frontier candidate exists that is better on "
        "every axis the regulator cares about."
    )
    lines.append("")
    lines.append(
        "The recommendation picks the frontier candidate with the highest "
        "overall weighted F1 subject to the configured inter-site "
        "fairness floor. This is the EU AI Act Art. 9 question: *which "
        "candidate minimises residual risk under the joint accuracy-and-"
        "fairness constraints we have declared?*"
    )
    lines.append("")
    return "\n".join(lines) + "\n"
