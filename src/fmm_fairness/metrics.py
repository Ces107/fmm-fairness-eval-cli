"""Fairness metrics for SaMD evaluation.

All metrics operate on a long-format pandas DataFrame with at minimum:
    - y_true:  int {0, 1}
    - y_pred:  int {0, 1}        (thresholded prediction)
    - y_score: float [0, 1]      (model probability / score)
    - <protected_attr>: str       (group identifier, e.g. "site_A", "F", "65+")

Conventions:
- "gap" is always max-over-groups minus min-over-groups (range), in [0, 1].
- Bootstrap CIs are percentile, BCa not implemented (small-sample caveat in README).
- Groups with fewer than `min_group_n` samples are excluded with a warning,
  not silently dropped (avoids spurious zero-gap from singletons).

References (defended in docs/samd-fairness-score.md):
- Hardt et al. 2016, "Equality of Opportunity in Supervised Learning", NeurIPS.
- Pierson et al. 2021, Nature Medicine 27:136-140.
- Char, Shah, Magnus 2018, NEJM 378:981-983.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MIN_GROUP_N_DEFAULT = 20
BOOTSTRAP_ITERS_DEFAULT = 1000
RNG_SEED_DEFAULT = 42


@dataclass
class GroupMetric:
    """Per-group result row."""
    group: str
    n: int
    value: float
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass
class FairnessResult:
    """Output of any fairness metric call."""
    metric_name: str
    attribute: str
    per_group: list[GroupMetric] = field(default_factory=list)
    gap: float = 0.0
    gap_ci_low: float | None = None
    gap_ci_high: float | None = None
    excluded_groups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "attribute": self.attribute,
            "gap": float(self.gap),
            "gap_ci_low": self.gap_ci_low,
            "gap_ci_high": self.gap_ci_high,
            "per_group": [
                {
                    "group": g.group,
                    "n": g.n,
                    "value": float(g.value),
                    "ci_low": g.ci_low,
                    "ci_high": g.ci_high,
                }
                for g in self.per_group
            ],
            "excluded_groups": self.excluded_groups,
        }


def _filter_groups(
    df: pd.DataFrame, attribute: str, min_group_n: int
) -> tuple[pd.DataFrame, list[str]]:
    """Drop groups with n < min_group_n. Return filtered df + excluded group names."""
    counts = df.groupby(attribute).size()
    keep = counts[counts >= min_group_n].index.tolist()
    excluded = counts[counts < min_group_n].index.tolist()
    if excluded:
        warnings.warn(
            f"Excluding {len(excluded)} groups of '{attribute}' with n < {min_group_n}: "
            f"{excluded}",
            stacklevel=2,
        )
    return df[df[attribute].isin(keep)].copy(), [str(g) for g in excluded]


def _true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TPR = TP / (TP + FN). Returns NaN if no positives."""
    pos = y_true == 1
    if pos.sum() == 0:
        return float("nan")
    return float((y_pred[pos] == 1).mean())


def _positive_prediction_rate(y_pred: np.ndarray) -> float:
    """P(y_pred = 1)."""
    if len(y_pred) == 0:
        return float("nan")
    return float((y_pred == 1).mean())


def _calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (ECE) with equal-width binning."""
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_score >= lo) & (y_score < hi) if i < n_bins - 1 else (y_score >= lo) & (y_score <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_score[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _bootstrap_gap(
    df: pd.DataFrame,
    attribute: str,
    per_group_fn: Callable[..., float],
    n_iters: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the max-min gap of per_group_fn applied per attribute group."""
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    groups = df[attribute].unique()
    indices_by_group = {g: df.index[df[attribute] == g].to_numpy() for g in groups}
    for _ in range(n_iters):
        vals = []
        for g in groups:
            idx = indices_by_group[g]
            sample_idx = rng.choice(idx, size=len(idx), replace=True)
            sample = df.loc[sample_idx]
            vals.append(per_group_fn(sample))
        vals_arr = np.array([v for v in vals if not np.isnan(v)])
        if len(vals_arr) < 2:
            continue
        gaps.append(float(vals_arr.max() - vals_arr.min()))
    if not gaps:
        return float("nan"), float("nan")
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def equal_opportunity_gap(
    df: pd.DataFrame,
    attribute: str,
    *,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    seed: int = RNG_SEED_DEFAULT,
) -> FairnessResult:
    """TPR-gap (Hardt et al. 2016): max-min TPR across protected groups."""
    df_f, excluded = _filter_groups(df, attribute, min_group_n)
    per_group: list[GroupMetric] = []
    tprs: list[float] = []
    for g, sub in df_f.groupby(attribute):
        tpr = _true_positive_rate(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        per_group.append(GroupMetric(group=str(g), n=len(sub), value=tpr))
        if not np.isnan(tpr):
            tprs.append(tpr)
    gap = (max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
    ci_low, ci_high = _bootstrap_gap(
        df_f,
        attribute,
        lambda s: _true_positive_rate(s["y_true"].to_numpy(), s["y_pred"].to_numpy()),
        bootstrap_iters,
        seed,
    )
    return FairnessResult(
        metric_name="equal_opportunity_gap",
        attribute=attribute,
        per_group=per_group,
        gap=gap,
        gap_ci_low=ci_low,
        gap_ci_high=ci_high,
        excluded_groups=excluded,
    )


def demographic_parity_gap(
    df: pd.DataFrame,
    attribute: str,
    *,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    seed: int = RNG_SEED_DEFAULT,
) -> FairnessResult:
    """P(y_pred=1)-gap across protected groups (Dwork et al. 2012 statistical parity)."""
    df_f, excluded = _filter_groups(df, attribute, min_group_n)
    per_group: list[GroupMetric] = []
    rates: list[float] = []
    for g, sub in df_f.groupby(attribute):
        r = _positive_prediction_rate(sub["y_pred"].to_numpy())
        per_group.append(GroupMetric(group=str(g), n=len(sub), value=r))
        if not np.isnan(r):
            rates.append(r)
    gap = (max(rates) - min(rates)) if len(rates) >= 2 else 0.0
    ci_low, ci_high = _bootstrap_gap(
        df_f,
        attribute,
        lambda s: _positive_prediction_rate(s["y_pred"].to_numpy()),
        bootstrap_iters,
        seed,
    )
    return FairnessResult(
        metric_name="demographic_parity_gap",
        attribute=attribute,
        per_group=per_group,
        gap=gap,
        gap_ci_low=ci_low,
        gap_ci_high=ci_high,
        excluded_groups=excluded,
    )


def calibration_gap(
    df: pd.DataFrame,
    attribute: str,
    *,
    n_bins: int = 10,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    seed: int = RNG_SEED_DEFAULT,
) -> FairnessResult:
    """ECE-gap across protected groups. Higher = more uneven calibration."""
    df_f, excluded = _filter_groups(df, attribute, min_group_n)
    per_group: list[GroupMetric] = []
    eces: list[float] = []
    for g, sub in df_f.groupby(attribute):
        ece = _calibration_error(
            sub["y_true"].to_numpy(), sub["y_score"].to_numpy(), n_bins=n_bins
        )
        per_group.append(GroupMetric(group=str(g), n=len(sub), value=ece))
        if not np.isnan(ece):
            eces.append(ece)
    gap = (max(eces) - min(eces)) if len(eces) >= 2 else 0.0
    ci_low, ci_high = _bootstrap_gap(
        df_f,
        attribute,
        lambda s: _calibration_error(
            s["y_true"].to_numpy(), s["y_score"].to_numpy(), n_bins=n_bins
        ),
        bootstrap_iters,
        seed,
    )
    return FairnessResult(
        metric_name="calibration_gap",
        attribute=attribute,
        per_group=per_group,
        gap=gap,
        gap_ci_low=ci_low,
        gap_ci_high=ci_high,
        excluded_groups=excluded,
    )


def inter_site_auc_variance(
    df: pd.DataFrame,
    site_attribute: str = "site",
    *,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
) -> FairnessResult:
    """Variance of per-site AUC. Single-number summary of inter-hospital generalization risk.

    NOTE: variance, not gap. We still expose it in the FairnessResult.gap field
    for serialization symmetry, but it is unbounded in [0, 0.25] (AUC variance).
    Per-group `value` field contains per-site AUC.
    """
    df_f, excluded = _filter_groups(df, site_attribute, min_group_n)
    per_group: list[GroupMetric] = []
    aucs: list[float] = []
    for g, sub in df_f.groupby(site_attribute):
        y_true = sub["y_true"].to_numpy()
        y_score = sub["y_score"].to_numpy()
        if len(np.unique(y_true)) < 2:
            per_group.append(GroupMetric(group=str(g), n=len(sub), value=float("nan")))
            continue
        auc = float(roc_auc_score(y_true, y_score))
        per_group.append(GroupMetric(group=str(g), n=len(sub), value=auc))
        aucs.append(auc)
    var = float(np.var(aucs)) if len(aucs) >= 2 else 0.0
    return FairnessResult(
        metric_name="inter_site_auc_variance",
        attribute=site_attribute,
        per_group=per_group,
        gap=var,
        excluded_groups=excluded,
    )


def samd_fairness_score(
    df: pd.DataFrame,
    site_attribute: str = "site",
    demographic_attributes: list[str] | None = None,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Composite SaMD fairness score in [0, 1] (higher = fairer).

    Formula (defended in docs/samd-fairness-score.md):
        F = 1 - clip( w_eo * EO + w_dp * DP + w_cal * CAL + w_site * SITE, 0, 1 )

    Where:
        EO     = mean equal-opportunity gap across declared demographic attrs
        DP     = mean demographic-parity gap across declared demographic attrs
        CAL    = mean calibration gap across declared demographic attrs
        SITE   = sqrt(inter_site_auc_variance) * 2   (rescale to [0,1])

    Default weights chosen from regulatory priority signal (SITE first, then EO):
        w_site=0.4, w_eo=0.3, w_dp=0.15, w_cal=0.15.

    Justification + sensitivity analysis: docs/samd-fairness-score.md.
    """
    if demographic_attributes is None:
        demographic_attributes = []
    w = {"site": 0.4, "eo": 0.3, "dp": 0.15, "cal": 0.15}
    if weights:
        w.update(weights)

    eo_gaps = [equal_opportunity_gap(df, a).gap for a in demographic_attributes]
    dp_gaps = [demographic_parity_gap(df, a).gap for a in demographic_attributes]
    cal_gaps = [calibration_gap(df, a).gap for a in demographic_attributes]

    eo_mean = float(np.mean(eo_gaps)) if eo_gaps else 0.0
    dp_mean = float(np.mean(dp_gaps)) if dp_gaps else 0.0
    cal_mean = float(np.mean(cal_gaps)) if cal_gaps else 0.0

    site_var = inter_site_auc_variance(df, site_attribute).gap
    site_term = min(1.0, float(np.sqrt(site_var)) * 2.0)

    raw = w["site"] * site_term + w["eo"] * eo_mean + w["dp"] * dp_mean + w["cal"] * cal_mean
    score = 1.0 - float(np.clip(raw, 0.0, 1.0))
    return {
        "samd_fairness_score": score,
        "components": {
            "site_term": site_term,
            "eo_mean": eo_mean,
            "dp_mean": dp_mean,
            "cal_mean": cal_mean,
        },
        "weights": w,
        "demographic_attributes": demographic_attributes,
        "site_attribute": site_attribute,
    }
