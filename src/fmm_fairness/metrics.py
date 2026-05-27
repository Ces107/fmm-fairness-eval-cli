"""Fairness metrics for SaMD evaluation.

Data model
----------

The minimum DataFrame shape is:

    y_true:  int in {0..K-1}
    y_pred:  int in {0..K-1}
    <protected_attr>: str            one column per declared attribute

For the score column there are two valid shapes:

    Binary (K = 2):
        y_score:  float in [0, 1]    P(class = 1)

    Multi-class (K >= 2):
        y_score_0, y_score_1, ..., y_score_{K-1}:  float, rows sum to ~1.0

A binary CSV using the multi-class shape (``y_score_0``, ``y_score_1``) is also
valid; the binary single-column shape stays the canonical form for backward
compatibility with v0.1 fixtures.

Conventions
-----------

- "gap" is always max-over-groups minus min-over-groups, in ``[0, 1]``.
- Per-group bootstrap CIs are percentile (BCa lands in S4 per the roadmap).
- Groups with fewer than ``min_group_n`` samples are excluded with a warning,
  never silently dropped (avoids spurious zero-gap from singletons).
- Equal-opportunity, demographic-parity, and calibration gaps are defined only
  for the binary case (``K = 2``). They raise a clear error on multi-class
  inputs. The roadmap parks the multi-class extensions of those criteria
  outside S1 because they require per-class operating-threshold choices that
  the CLI does not yet ingest.

References
----------

- Hardt, Price, Srebro 2016, "Equality of Opportunity in Supervised Learning",
  NeurIPS.
- Dwork, Hardt, Pitassi, Reingold, Zemel 2012, "Fairness Through Awareness",
  ITCS.
- Pleiss, Raghavan, Wu, Kleinberg, Weinberger 2017, "On Fairness and
  Calibration", NeurIPS.
- Pierson et al. 2021, Nature Medicine 27:136-140.
- Seyyed-Kalantari et al. 2021, Nature Medicine 27:2176-2182.
- Lu et al. 2024 ("CONCH"), Nature Medicine 30:863-874.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from fmm_fairness.statistics import _coerce_bracket

MIN_GROUP_N_DEFAULT = 20
BOOTSTRAP_ITERS_DEFAULT = 1000
RNG_SEED_DEFAULT = 42

SCORE_COL_BINARY = "y_score"
SCORE_COL_PREFIX = "y_score_"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GroupMetric:
    """Per-group result row.

    ``per_class`` is populated only by per-class metrics; for scalar metrics
    it stays None and is omitted from the serialised dict.
    """

    group: str
    n: int
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    per_class: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "group": self.group,
            "n": self.n,
            "value": float(self.value),
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }
        if self.per_class is not None:
            d["per_class"] = [float(v) for v in self.per_class]
        return d


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
    per_class_gap: list[float] | None = None  # populated by per_class_f1_gap
    bootstrap_method: str | None = None       # "bca" | "percentile"
    bootstrap_se: float | None = None
    permutation_p_value: float | None = None
    permutation_iters: int | None = None
    minimum_detectable_effect: float | None = None
    alpha: float | None = None
    power: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "metric": self.metric_name,
            "attribute": self.attribute,
            "gap": float(self.gap),
            "gap_ci_low": self.gap_ci_low,
            "gap_ci_high": self.gap_ci_high,
            "per_group": [g.to_dict() for g in self.per_group],
            "excluded_groups": self.excluded_groups,
        }
        if self.per_class_gap is not None:
            d["per_class_gap"] = [float(v) for v in self.per_class_gap]
        if self.bootstrap_method is not None:
            d["bootstrap_method"] = self.bootstrap_method
        if self.bootstrap_se is not None and not np.isnan(self.bootstrap_se):
            d["bootstrap_se"] = float(self.bootstrap_se)
        if self.permutation_p_value is not None:
            d["permutation_p_value"] = float(self.permutation_p_value)
            d["permutation_iters"] = self.permutation_iters
        if self.minimum_detectable_effect is not None and not np.isnan(
            self.minimum_detectable_effect
        ):
            d["minimum_detectable_effect"] = float(self.minimum_detectable_effect)
            d["alpha"] = self.alpha
            d["power"] = self.power
        return d


# ---------------------------------------------------------------------------
# Score / class-count detection
# ---------------------------------------------------------------------------


def _multi_class_score_columns(df: pd.DataFrame) -> list[str]:
    """Return ``y_score_0..y_score_{K-1}`` columns sorted by class index, or []."""
    multi = []
    for c in df.columns:
        if not c.startswith(SCORE_COL_PREFIX):
            continue
        suffix = c[len(SCORE_COL_PREFIX):]
        if suffix.isdigit():
            multi.append((int(suffix), c))
    multi.sort(key=lambda t: t[0])
    if not multi:
        return []
    indices = [i for i, _ in multi]
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(
            "Multi-class score columns must form a contiguous sequence "
            f"y_score_0..y_score_{{K-1}}; got indices {indices}.",
        )
    return [name for _, name in multi]


def detect_num_classes(
    df: pd.DataFrame, num_classes: int | None = None
) -> int:
    """Detect K from score columns and labels; cross-check against override.

    Resolution order:
        1. If ``num_classes`` was passed explicitly, it wins, but inconsistency
           with the score columns or with ``max(y_true)+1`` raises a warning.
        2. Otherwise, infer from ``y_score_0..y_score_{K-1}`` columns if
           present; else from the binary ``y_score`` column; else from
           ``max(y_true)+1``.
    """
    multi_cols = _multi_class_score_columns(df)
    if multi_cols:
        k_detected = len(multi_cols)
    elif SCORE_COL_BINARY in df.columns:
        k_detected = 2
    elif "y_true" in df.columns:
        k_detected = int(df["y_true"].max()) + 1
    else:
        raise ValueError("Cannot detect number of classes: no y_score* or y_true columns.")

    if num_classes is None:
        return k_detected
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2; got {num_classes}.")
    if num_classes != k_detected:
        warnings.warn(
            f"num_classes={num_classes} disagrees with detected K={k_detected} "
            f"(score columns / y_true range); using the explicit value.",
            stacklevel=2,
        )
    return num_classes


def _score_matrix(df: pd.DataFrame, num_classes: int) -> np.ndarray:
    """Return a (n, K) probability matrix.

    Binary single-column shape is lifted to ``[1 - p, p]``. Multi-class
    ``y_score_k`` columns are stacked in class order.
    """
    expected_cols = [f"{SCORE_COL_PREFIX}{k}" for k in range(num_classes)]
    if all(c in df.columns for c in expected_cols):
        return np.asarray(df[expected_cols].to_numpy(dtype=float))
    if num_classes == 2 and SCORE_COL_BINARY in df.columns:
        p = np.asarray(df[SCORE_COL_BINARY].to_numpy(dtype=float))
        return np.stack([1.0 - p, p], axis=1)
    raise ValueError(
        f"Expected score columns {expected_cols} or binary 'y_score' for K=2; "
        f"got columns {[c for c in df.columns if c.startswith(SCORE_COL_PREFIX) or c == SCORE_COL_BINARY]}.",
    )


def _binary_score_array(df: pd.DataFrame) -> np.ndarray:
    """Return the canonical binary probability vector P(class=1) for K=2 inputs."""
    if SCORE_COL_BINARY in df.columns and not _multi_class_score_columns(df):
        return np.asarray(df[SCORE_COL_BINARY].to_numpy(dtype=float))
    return np.asarray(_score_matrix(df, 2)[:, 1])


# ---------------------------------------------------------------------------
# Group filtering and bootstrap
# ---------------------------------------------------------------------------


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


def _bootstrap_gap(
    df: pd.DataFrame,
    attribute: str,
    per_group_fn: Callable[..., float],
    n_iters: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the max-min gap of ``per_group_fn`` over groups.

    The CI is bracket-coerced to contain the observed (non-bootstrap) gap:
    the max-min statistic is positively biased, so a near-zero true gap can
    yield a percentile interval that sits entirely above the point estimate.
    See :func:`fmm_fairness.statistics._coerce_bracket`.
    """
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    groups = df[attribute].unique()
    indices_by_group = {g: df.index[df[attribute] == g].to_numpy() for g in groups}

    # Observed (non-bootstrap) gap = the point estimate the report shows.
    obs_vals = [
        per_group_fn(df.loc[indices_by_group[g]]) for g in groups
    ]
    obs_arr = np.array([v for v in obs_vals if not np.isnan(v)])
    theta_hat = float(obs_arr.max() - obs_arr.min()) if len(obs_arr) >= 2 else float("nan")

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
    lo, hi = float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))
    if not np.isnan(theta_hat):
        lo, hi = _coerce_bracket(theta_hat, lo, hi)
    return lo, hi


def _bootstrap_per_class_worst_gap(
    df: pd.DataFrame,
    attribute: str,
    num_classes: int,
    n_iters: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the worst-per-class F1 gap across groups.

    Bracket-coerced to contain the observed worst-class gap (same positive-
    bias rationale as :func:`_bootstrap_gap`).
    """
    rng = np.random.default_rng(seed)
    worst: list[float] = []
    groups = df[attribute].unique()
    indices_by_group = {g: df.index[df[attribute] == g].to_numpy() for g in groups}

    # Observed (non-bootstrap) worst-class gap = the point estimate.
    obs_vecs = [
        _per_class_f1(
            df.loc[indices_by_group[g], "y_true"].to_numpy(),
            df.loc[indices_by_group[g], "y_pred"].to_numpy(),
            num_classes,
        )
        for g in groups
    ]
    if len(obs_vecs) >= 2:
        obs_stack = np.stack(obs_vecs, axis=0)
        theta_hat = float((obs_stack.max(axis=0) - obs_stack.min(axis=0)).max())
    else:
        theta_hat = float("nan")

    for _ in range(n_iters):
        per_group_vecs: list[np.ndarray] = []
        for g in groups:
            idx = indices_by_group[g]
            sample_idx = rng.choice(idx, size=len(idx), replace=True)
            sample = df.loc[sample_idx]
            per_group_vecs.append(
                _per_class_f1(
                    sample["y_true"].to_numpy(),
                    sample["y_pred"].to_numpy(),
                    num_classes,
                )
            )
        if len(per_group_vecs) < 2:
            continue
        stacked = np.stack(per_group_vecs, axis=0)  # (G, K)
        per_class_gap = stacked.max(axis=0) - stacked.min(axis=0)
        worst.append(float(per_class_gap.max()))
    if not worst:
        return float("nan"), float("nan")
    lo, hi = float(np.percentile(worst, 2.5)), float(np.percentile(worst, 97.5))
    if not np.isnan(theta_hat):
        lo, hi = _coerce_bracket(theta_hat, lo, hi)
    return lo, hi


# ---------------------------------------------------------------------------
# Per-sample helper metrics
# ---------------------------------------------------------------------------


def _true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true == 1
    if pos.sum() == 0:
        return float("nan")
    return float((y_pred[pos] == 1).mean())


def _positive_prediction_rate(y_pred: np.ndarray) -> float:
    if len(y_pred) == 0:
        return float("nan")
    return float((y_pred == 1).mean())


def _calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_score >= lo) & (y_score < hi)
        else:
            mask = (y_score >= lo) & (y_score <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_score[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Support-weighted F1 over all K classes."""
    if len(y_true) == 0:
        return float("nan")
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="weighted",
            zero_division=0,
        )
    )


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro F1 over all K classes (equal class weight)."""
    if len(y_true) == 0:
        return float("nan")
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
    )


def _per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Return a length-K array of per-class F1 scores. Missing classes => 0.0."""
    if len(y_true) == 0:
        return np.full(num_classes, np.nan, dtype=float)
    return np.asarray(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average=None,
            zero_division=0,
        ),
        dtype=float,
    )


def _group_ovr_macro_auc(
    y_true: np.ndarray,
    score_matrix: np.ndarray,
    num_classes: int,
) -> float:
    """OVR macro AUC; NaN if fewer than 2 classes present in y_true."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    if num_classes == 2:
        return float(roc_auc_score(y_true, score_matrix[:, 1]))
    try:
        return float(
            roc_auc_score(
                y_true,
                score_matrix,
                multi_class="ovr",
                average="macro",
                labels=list(range(num_classes)),
            )
        )
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Binary-only fairness metrics (unchanged numerics from v0.1)
# ---------------------------------------------------------------------------


def _require_binary(df: pd.DataFrame, metric_name: str) -> None:
    """Raise a clear error if the data is not binary."""
    K = detect_num_classes(df)
    if K != 2:
        raise ValueError(
            f"{metric_name} is defined only for binary outcomes (K=2); detected K={K}. "
            f"For multi-class data use weighted_f1_gap / macro_f1_gap / per_class_f1_gap.",
        )
    if not set(df["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{metric_name}: y_true must be in {{0, 1}}.")


def equal_opportunity_gap(
    df: pd.DataFrame,
    attribute: str,
    *,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    seed: int = RNG_SEED_DEFAULT,
) -> FairnessResult:
    """TPR-gap (Hardt et al. 2016). Binary only."""
    _require_binary(df, "equal_opportunity_gap")
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
    """P(y_pred=1)-gap across groups (Dwork et al. 2012). Binary only."""
    _require_binary(df, "demographic_parity_gap")
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
    """ECE-gap across groups (Pleiss et al. 2017). Binary only."""
    _require_binary(df, "calibration_gap")
    y_score_arr = _binary_score_array(df)
    df_work = df.copy()
    df_work[SCORE_COL_BINARY] = y_score_arr  # ensures bootstrap uses the canonical column
    df_f, excluded = _filter_groups(df_work, attribute, min_group_n)
    per_group: list[GroupMetric] = []
    eces: list[float] = []
    for g, sub in df_f.groupby(attribute):
        ece = _calibration_error(
            sub["y_true"].to_numpy(), sub[SCORE_COL_BINARY].to_numpy(), n_bins=n_bins
        )
        per_group.append(GroupMetric(group=str(g), n=len(sub), value=ece))
        if not np.isnan(ece):
            eces.append(ece)
    gap = (max(eces) - min(eces)) if len(eces) >= 2 else 0.0
    ci_low, ci_high = _bootstrap_gap(
        df_f,
        attribute,
        lambda s: _calibration_error(
            s["y_true"].to_numpy(), s[SCORE_COL_BINARY].to_numpy(), n_bins=n_bins
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


# ---------------------------------------------------------------------------
# Multi-class-aware F1 family (new in v0.2)
# ---------------------------------------------------------------------------


def _attach_inference(
    result: FairnessResult,
    df_f: pd.DataFrame,
    attribute: str,
    per_group_fn: Callable[[pd.DataFrame], float],
    *,
    bootstrap_method: str,
    bootstrap_iters: int,
    permutation_iters: int,
    alpha: float,
    power: float,
    seed: int,
) -> FairnessResult:
    """Attach BCa/percentile CI + optional permutation p-value + MDE to a result."""
    from fmm_fairness.statistics import gap_inference

    inf = gap_inference(
        df_f,
        attribute,
        per_group_fn,
        bootstrap_method=bootstrap_method,
        n_bootstrap_iters=bootstrap_iters,
        n_permutation_iters=permutation_iters,
        alpha=alpha,
        power=power,
        seed=seed,
    )
    result.gap_ci_low = inf.ci_low
    result.gap_ci_high = inf.ci_high
    result.bootstrap_method = inf.bootstrap_method
    result.bootstrap_se = inf.bootstrap_se
    result.permutation_p_value = inf.permutation_p_value
    result.permutation_iters = inf.permutation_iters
    result.minimum_detectable_effect = inf.minimum_detectable_effect
    result.alpha = inf.alpha
    result.power = inf.power
    return result


def weighted_f1_gap(
    df: pd.DataFrame,
    attribute: str,
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    bootstrap_method: str = "bca",
    permutation_iters: int = 0,
    alpha: float = 0.05,
    power: float = 0.80,
    seed: int = RNG_SEED_DEFAULT,
) -> FairnessResult:
    """Support-weighted F1 gap across protected groups.

    This is the headline inter-site fairness metric of the underlying TFG.
    Defined for any K >= 2 (binary or multi-class). Default CI method is
    BCa (Efron 1987); permutation p-value is off by default to keep the
    runtime under the test bootstrap budget, opt-in via ``permutation_iters``.
    """
    K = detect_num_classes(df, num_classes)
    df_f, excluded = _filter_groups(df, attribute, min_group_n)
    per_group: list[GroupMetric] = []
    f1s: list[float] = []
    for g, sub in df_f.groupby(attribute):
        f1 = _weighted_f1(
            sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), K
        )
        per_group.append(GroupMetric(group=str(g), n=len(sub), value=f1))
        if not np.isnan(f1):
            f1s.append(f1)
    gap = (max(f1s) - min(f1s)) if len(f1s) >= 2 else 0.0
    result = FairnessResult(
        metric_name="weighted_f1_gap",
        attribute=attribute,
        per_group=per_group,
        gap=gap,
        excluded_groups=excluded,
    )
    return _attach_inference(
        result,
        df_f,
        attribute,
        lambda s: _weighted_f1(s["y_true"].to_numpy(), s["y_pred"].to_numpy(), K),
        bootstrap_method=bootstrap_method,
        bootstrap_iters=bootstrap_iters,
        permutation_iters=permutation_iters,
        alpha=alpha,
        power=power,
        seed=seed,
    )


def macro_f1_gap(
    df: pd.DataFrame,
    attribute: str,
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    bootstrap_method: str = "bca",
    permutation_iters: int = 0,
    alpha: float = 0.05,
    power: float = 0.80,
    seed: int = RNG_SEED_DEFAULT,
) -> FairnessResult:
    """Macro F1 gap across protected groups (equal class weight)."""
    K = detect_num_classes(df, num_classes)
    df_f, excluded = _filter_groups(df, attribute, min_group_n)
    per_group: list[GroupMetric] = []
    f1s: list[float] = []
    for g, sub in df_f.groupby(attribute):
        f1 = _macro_f1(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), K)
        per_group.append(GroupMetric(group=str(g), n=len(sub), value=f1))
        if not np.isnan(f1):
            f1s.append(f1)
    gap = (max(f1s) - min(f1s)) if len(f1s) >= 2 else 0.0
    result = FairnessResult(
        metric_name="macro_f1_gap",
        attribute=attribute,
        per_group=per_group,
        gap=gap,
        excluded_groups=excluded,
    )
    return _attach_inference(
        result,
        df_f,
        attribute,
        lambda s: _macro_f1(s["y_true"].to_numpy(), s["y_pred"].to_numpy(), K),
        bootstrap_method=bootstrap_method,
        bootstrap_iters=bootstrap_iters,
        permutation_iters=permutation_iters,
        alpha=alpha,
        power=power,
        seed=seed,
    )


def per_class_f1_gap(
    df: pd.DataFrame,
    attribute: str,
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    seed: int = RNG_SEED_DEFAULT,
) -> FairnessResult:
    """Per-class F1 gap across groups.

    Returns a FairnessResult whose:
        - ``per_group[i].per_class`` carries the length-K F1 vector for group i.
        - ``per_class_gap`` is the length-K array of max-min per-class gaps.
        - ``gap`` is the worst (max) entry of ``per_class_gap``: the
          single-class disparity that drives the headline.
        - ``gap_ci_low/high`` are a bootstrap CI on the worst-class gap.
    """
    K = detect_num_classes(df, num_classes)
    df_f, excluded = _filter_groups(df, attribute, min_group_n)
    per_group: list[GroupMetric] = []
    vectors: list[np.ndarray] = []
    for g, sub in df_f.groupby(attribute):
        vec = _per_class_f1(
            sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), K
        )
        # Weighted F1 is a useful scalar summary on the per-group row.
        scalar = _weighted_f1(
            sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), K
        )
        per_group.append(
            GroupMetric(
                group=str(g),
                n=len(sub),
                value=scalar,
                per_class=[float(v) for v in vec],
            )
        )
        vectors.append(vec)
    if len(vectors) >= 2:
        stacked = np.stack(vectors, axis=0)  # (G, K)
        per_class_gap_arr = (stacked.max(axis=0) - stacked.min(axis=0)).astype(float)
        worst_gap = float(per_class_gap_arr.max())
    else:
        per_class_gap_arr = np.zeros(K, dtype=float)
        worst_gap = 0.0
    ci_low, ci_high = _bootstrap_per_class_worst_gap(
        df_f, attribute, K, bootstrap_iters, seed
    )
    return FairnessResult(
        metric_name="per_class_f1_gap",
        attribute=attribute,
        per_group=per_group,
        gap=worst_gap,
        gap_ci_low=ci_low,
        gap_ci_high=ci_high,
        excluded_groups=excluded,
        per_class_gap=per_class_gap_arr.tolist(),
    )


def multi_class_auc_gap(
    df: pd.DataFrame,
    attribute: str,
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    seed: int = RNG_SEED_DEFAULT,
) -> FairnessResult:
    """Max-min OVR-macro AUC across groups (binary AUC if K=2)."""
    K = detect_num_classes(df, num_classes)
    df_f, excluded = _filter_groups(df, attribute, min_group_n)
    per_group: list[GroupMetric] = []
    aucs: list[float] = []
    for g, sub in df_f.groupby(attribute):
        score_mat = _score_matrix(sub, K)
        auc = _group_ovr_macro_auc(sub["y_true"].to_numpy(), score_mat, K)
        per_group.append(GroupMetric(group=str(g), n=len(sub), value=auc))
        if not np.isnan(auc):
            aucs.append(auc)
    gap = (max(aucs) - min(aucs)) if len(aucs) >= 2 else 0.0

    def _resample_fn(s: pd.DataFrame) -> float:
        return _group_ovr_macro_auc(
            s["y_true"].to_numpy(), _score_matrix(s, K), K
        )

    ci_low, ci_high = _bootstrap_gap(
        df_f, attribute, _resample_fn, bootstrap_iters, seed
    )
    return FairnessResult(
        metric_name="multi_class_auc_gap",
        attribute=attribute,
        per_group=per_group,
        gap=gap,
        gap_ci_low=ci_low,
        gap_ci_high=ci_high,
        excluded_groups=excluded,
    )


# ---------------------------------------------------------------------------
# Site-level metric (K-aware)
# ---------------------------------------------------------------------------


def inter_site_auc_variance(
    df: pd.DataFrame,
    site_attribute: str = "site",
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
) -> FairnessResult:
    """Variance of per-site AUC. K-aware: OVR-macro AUC for K>2, binary AUC for K=2.

    The ``gap`` field carries the variance for serialisation symmetry with the
    other metrics; semantically it is unbounded above by ~0.25.
    Per-group ``value`` is per-site AUC.
    """
    K = detect_num_classes(df, num_classes)
    df_f, excluded = _filter_groups(df, site_attribute, min_group_n)
    per_group: list[GroupMetric] = []
    aucs: list[float] = []
    for g, sub in df_f.groupby(site_attribute):
        y_true = sub["y_true"].to_numpy()
        if K == 2:
            # Bit-identical to v0.1 when the canonical binary score column is present.
            y_score = _binary_score_array(sub)
            if len(np.unique(y_true)) < 2:
                per_group.append(GroupMetric(group=str(g), n=len(sub), value=float("nan")))
                continue
            auc = float(roc_auc_score(y_true, y_score))
        else:
            score_mat = _score_matrix(sub, K)
            auc = _group_ovr_macro_auc(y_true, score_mat, K)
            if np.isnan(auc):
                per_group.append(GroupMetric(group=str(g), n=len(sub), value=float("nan")))
                continue
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


# ---------------------------------------------------------------------------
# Composite SaMD fairness score
# ---------------------------------------------------------------------------


def samd_fairness_score(
    df: pd.DataFrame,
    site_attribute: str = "site",
    demographic_attributes: list[str] | None = None,
    *,
    num_classes: int | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Composite SaMD fairness score in [0, 1] (higher = fairer).

    v0.2 formula (defended in docs/samd-fairness-score.md):

        F = 1 - clip(
            w_f1_site * F1_SITE
          + w_site    * SITE
          + w_eo      * EO
          + w_dp      * DP
          + w_cal     * CAL,
            0, 1,
        )

    where:
        F1_SITE = weighted_f1_gap at the site attribute (the TFG headline)
        SITE    = min(1, 2 * sqrt(Var_{g in s}(AUC(g))))  (K-aware AUC)
        EO/DP/CAL = mean of the corresponding gap over declared demographic
                    attributes; defined only for K=2 (computed and contributed
                    when K=2; force-zeroed and excluded from weights re-norm
                    when K > 2, since the underlying criteria require binary
                    operating thresholds the CLI does not yet ingest).

    Default weights (sum to 1.0):
        w_f1_site = 0.35  (TFG-headline disparity, dominant inter-site term)
        w_site    = 0.20  (AUC-variance generalisation risk)
        w_eo      = 0.20  (TPR disparity, binary only)
        w_dp      = 0.10  (selection-rate disparity, binary only)
        w_cal     = 0.15  (calibration disparity, binary only)

    When K > 2 and the binary-only terms are not computable, the remaining
    weights are renormalised over {F1_SITE, SITE} to preserve the [0,1] range
    of the deduction without distorting the relative weighting between the
    two terms that are defined for any K.
    """
    if demographic_attributes is None:
        demographic_attributes = []
    K = detect_num_classes(df, num_classes)

    default_w = {
        "f1_site": 0.35,
        "site": 0.20,
        "eo": 0.20,
        "dp": 0.10,
        "cal": 0.15,
    }
    w = dict(default_w)
    if weights:
        w.update(weights)

    f1_site_term = 0.0
    if site_attribute in df.columns:
        f1_site_term = float(
            np.clip(weighted_f1_gap(df, site_attribute, num_classes=K).gap, 0.0, 1.0)
        )

    site_var = (
        inter_site_auc_variance(df, site_attribute, num_classes=K).gap
        if site_attribute in df.columns
        else 0.0
    )
    site_term = min(1.0, float(np.sqrt(site_var)) * 2.0)

    if K == 2:
        eo_gaps = [equal_opportunity_gap(df, a).gap for a in demographic_attributes]
        dp_gaps = [demographic_parity_gap(df, a).gap for a in demographic_attributes]
        cal_gaps = [calibration_gap(df, a).gap for a in demographic_attributes]
        eo_mean = float(np.mean(eo_gaps)) if eo_gaps else 0.0
        dp_mean = float(np.mean(dp_gaps)) if dp_gaps else 0.0
        cal_mean = float(np.mean(cal_gaps)) if cal_gaps else 0.0
        effective_w = w
    else:
        eo_mean = dp_mean = cal_mean = 0.0
        # Renormalise {f1_site, site} so the deduction still ranges over [0,1].
        active = {"f1_site": w["f1_site"], "site": w["site"]}
        s = sum(active.values())
        if s > 0:
            scaled = {k: v / s for k, v in active.items()}
        else:
            scaled = {"f1_site": 0.5, "site": 0.5}
        effective_w = {**w, **scaled, "eo": 0.0, "dp": 0.0, "cal": 0.0}

    raw = (
        effective_w["f1_site"] * f1_site_term
        + effective_w["site"] * site_term
        + effective_w["eo"] * eo_mean
        + effective_w["dp"] * dp_mean
        + effective_w["cal"] * cal_mean
    )
    score = 1.0 - float(np.clip(raw, 0.0, 1.0))
    return {
        "samd_fairness_score": score,
        "components": {
            "f1_site_term": f1_site_term,
            "site_term": site_term,
            "eo_mean": eo_mean,
            "dp_mean": dp_mean,
            "cal_mean": cal_mean,
        },
        "weights": effective_w,
        "weights_declared": w,
        "demographic_attributes": demographic_attributes,
        "site_attribute": site_attribute,
        "num_classes": K,
    }
