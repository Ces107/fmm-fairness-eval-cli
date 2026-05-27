"""Calibration depth — Brier score, Hosmer-Lemeshow test, reliability bins.

Calibration sits next to discrimination (F1, AUC) in the SaMD evidence
package. A model that is highly accurate but poorly calibrated produces
miscalibrated risk scores that downstream decision rules trust as
probabilities. The AI Act Art. 15 robustness requirement is most often
read as discrimination plus calibration jointly.

Public API
----------
- ``brier_score(y_true, score_matrix, num_classes)`` -> scalar
- ``per_class_brier(y_true, score_matrix, num_classes)`` -> length-K array
- ``hosmer_lemeshow(y_true, y_score, n_bins=10)`` -> dict(chi2, df, p_value, bins)
- ``reliability_bins(y_true, y_score, n_bins=10)`` -> dict(bin_centers, accuracies, confidences, counts)
- ``per_group_calibration(df, attribute, num_classes, ...)`` -> per-group block
- ``build_calibration_block(df, cfg)`` -> evidence-pack-ready dict

Conventions
-----------
- Brier score is the multi-class extension (Brier 1950 / Murphy 1973):
  ``mean over samples of  sum_k (y_onehot_k - p_k)^2``. For K=2 this matches
  the standard binary Brier ``mean((y - p)^2)`` (up to a factor of 2).
- Hosmer-Lemeshow is the C-statistic with equal-frequency deciles. It is
  defined only for binary tasks; multi-class deployments compute a one-vs-rest
  HL per class. Degrees-of-freedom ``g - 2`` is the standard model-evaluation
  convention.
- Reliability bins use equal-width [0, 1] bins (10 by default). The
  ``--render-plots`` flag downstream paints these bins onto a PNG; the JSON
  evidence always carries the numeric bin data, plot or no plot.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from fmm_fairness.metrics import (
    MIN_GROUP_N_DEFAULT,
    _score_matrix,
    detect_num_classes,
)

HL_DEFAULT_BINS = 10
RELIABILITY_DEFAULT_BINS = 10


def _validate_y_true_range(y_true: np.ndarray, num_classes: int) -> None:
    """Public-API guard: y_true integer labels must lie in [0, num_classes).

    Without this guard, the one-hot construction
    ``onehot[arange(n), y_true.astype(int)] = 1`` raises ``IndexError`` with
    a NumPy-internal message (TD-003). Surface a clear ``ValueError`` instead.
    """
    if len(y_true) == 0:
        return
    y_int = y_true.astype(int)
    bad_high = int((y_int >= num_classes).sum())
    bad_low = int((y_int < 0).sum())
    if bad_high or bad_low:
        raise ValueError(
            f"y_true contains {bad_high + bad_low} labels outside "
            f"[0, {num_classes}); got min={int(y_int.min())} max={int(y_int.max())}."
        )


def brier_score(
    y_true: np.ndarray, score_matrix: np.ndarray, num_classes: int
) -> float:
    """Multi-class Brier score on a (n, K) probability matrix.

    Returns NaN for an empty input. The score is in ``[0, 2]`` (worst is the
    matrix that assigns probability 1 to the wrong class on every sample).
    """
    if len(y_true) == 0:
        return float("nan")
    if score_matrix.shape[1] != num_classes:
        raise ValueError(
            f"score_matrix has {score_matrix.shape[1]} columns; expected {num_classes}."
        )
    _validate_y_true_range(y_true, num_classes)
    onehot = np.zeros_like(score_matrix, dtype=float)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(((onehot - score_matrix) ** 2).sum(axis=1).mean())


def per_class_brier(
    y_true: np.ndarray, score_matrix: np.ndarray, num_classes: int
) -> np.ndarray:
    """Per-class Brier vector: ``mean_n ((y == k) - p_k)^2`` for k = 0..K-1."""
    if len(y_true) == 0:
        return np.full(num_classes, np.nan, dtype=float)
    if score_matrix.shape[1] != num_classes:
        raise ValueError(
            f"score_matrix has {score_matrix.shape[1]} columns; expected {num_classes}."
        )
    _validate_y_true_range(y_true, num_classes)
    onehot = np.zeros_like(score_matrix, dtype=float)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return np.asarray(((onehot - score_matrix) ** 2).mean(axis=0), dtype=float)


def hosmer_lemeshow(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = HL_DEFAULT_BINS
) -> dict[str, Any]:
    """Hosmer-Lemeshow chi-square goodness-of-fit test (binary).

    Returns a dict with ``chi2``, ``df``, ``p_value``, ``n_bins`` (effective),
    and a list of per-bin ``(n, observed_pos, expected_pos)`` tuples.

    Edge cases:
    - Bins with n < 2 are merged into the previous bin so the chi-square sum
      stays defined; the effective ``n_bins`` returned can be < ``n_bins``.
    - Bins whose expected count is degenerate (``E*(n-E)/n`` = 0) are skipped.
    - With fewer than 2 retained bins the test returns NaN; this surfaces
      in the evidence pack with a clear flag rather than a false significance.
    """
    if len(y_true) == 0:
        return {
            "chi2": float("nan"),
            "df": 0,
            "p_value": float("nan"),
            "n_bins": 0,
            "bins": [],
            "note": "empty input",
        }
    if len(np.unique(y_true)) < 2:
        return {
            "chi2": float("nan"),
            "df": 0,
            "p_value": float("nan"),
            "n_bins": 0,
            "bins": [],
            "note": "y_true is constant; HL undefined",
        }
    order = np.argsort(y_score)
    y_true_sorted = y_true[order].astype(int)
    y_score_sorted = y_score[order].astype(float)
    bin_edges = np.array_split(np.arange(len(y_true_sorted)), n_bins)
    bins: list[tuple[int, int, float]] = []
    for idxs in bin_edges:
        if len(idxs) == 0:
            continue
        n_g = len(idxs)
        observed = int(y_true_sorted[idxs].sum())
        expected = float(y_score_sorted[idxs].sum())
        bins.append((n_g, observed, expected))
    # Merge any too-small bin into its neighbour. Earlier versions only merged
    # backward (into the predecessor), which silently skipped a too-small FIRST
    # bin because there was no predecessor to merge into (TD-004). Now a small
    # first bin is "pending" and merges forward into the next bin instead.
    cleaned: list[tuple[int, int, float]] = []
    pending: tuple[int, int, float] | None = None
    for bin_ in bins:
        if pending is not None:
            bin_ = (
                pending[0] + bin_[0],
                pending[1] + bin_[1],
                pending[2] + bin_[2],
            )
            pending = None
        if cleaned and bin_[0] < 2:
            prev = cleaned[-1]
            cleaned[-1] = (prev[0] + bin_[0], prev[1] + bin_[1], prev[2] + bin_[2])
        elif not cleaned and bin_[0] < 2:
            pending = bin_  # defer to next bin (forward-merge)
        else:
            cleaned.append(bin_)
    if pending is not None and cleaned:
        # Every subsequent bin was already absorbed; fold the pending head
        # into the now-last bin so its mass is not lost.
        prev = cleaned[-1]
        cleaned[-1] = (
            prev[0] + pending[0],
            prev[1] + pending[1],
            prev[2] + pending[2],
        )
    chi2_sum = 0.0
    valid_bins = 0
    for n_g, o, e in cleaned:
        if n_g <= 0:
            continue
        denom = e * (1.0 - e / n_g)
        if denom <= 0:
            continue
        chi2_sum += (o - e) ** 2 / denom
        valid_bins += 1
    if valid_bins < 2:
        return {
            "chi2": float("nan"),
            "df": 0,
            "p_value": float("nan"),
            "n_bins": valid_bins,
            "bins": [{"n": n, "observed": o, "expected": e} for n, o, e in cleaned],
            "note": "insufficient non-degenerate bins",
        }
    df = max(valid_bins - 2, 1)
    p_value = float(1.0 - _scipy_stats.chi2.cdf(chi2_sum, df=df))
    return {
        "chi2": float(chi2_sum),
        "df": int(df),
        "p_value": p_value,
        "n_bins": valid_bins,
        "bins": [{"n": n, "observed": o, "expected": e} for n, o, e in cleaned],
        "note": None,
    }


def reliability_bins(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = RELIABILITY_DEFAULT_BINS,
) -> dict[str, Any]:
    """Equal-width reliability-diagram bins on a binary probability vector.

    Returns ``bin_centers`` (length n_bins), and four parallel arrays:
    ``bin_lows``, ``bin_highs``, ``counts``, ``confidences`` (mean predicted
    probability per bin), ``accuracies`` (mean observed positive rate per bin).
    Empty bins emit NaN for ``confidence`` and ``accuracy`` and 0 for ``count``.
    """
    if len(y_true) == 0:
        return {
            "n_bins": n_bins,
            "bin_lows": [],
            "bin_highs": [],
            "bin_centers": [],
            "counts": [],
            "confidences": [],
            "accuracies": [],
        }
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    lows = edges[:-1].tolist()
    highs = edges[1:].tolist()
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    counts: list[int] = []
    confidences: list[float] = []
    accuracies: list[float] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (y_score >= lo) & (y_score < hi)
        else:
            mask = (y_score >= lo) & (y_score <= hi)
        n = int(mask.sum())
        counts.append(n)
        if n == 0:
            confidences.append(float("nan"))
            accuracies.append(float("nan"))
        else:
            confidences.append(float(y_score[mask].mean()))
            accuracies.append(float((y_true[mask] == 1).mean()))
    return {
        "n_bins": n_bins,
        "bin_lows": lows,
        "bin_highs": highs,
        "bin_centers": centers,
        "counts": counts,
        "confidences": confidences,
        "accuracies": accuracies,
    }


def _binary_score_for_class(
    score_matrix: np.ndarray, target_class: int
) -> np.ndarray:
    """Return P(class = target_class) as a flat array; one-vs-rest view."""
    return np.asarray(score_matrix[:, target_class], dtype=float)


def _ovr_y_true(y_true: np.ndarray, target_class: int) -> np.ndarray:
    return np.asarray(y_true == target_class, dtype=int)


def per_group_calibration(
    df: pd.DataFrame,
    attribute: str,
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    n_reliability_bins: int = RELIABILITY_DEFAULT_BINS,
    n_hl_bins: int = HL_DEFAULT_BINS,
) -> dict[str, Any]:
    """Per-group calibration block.

    For each group with ``n >= min_group_n`` returns:
        - Brier score (multi-class)
        - Per-class Brier vector
        - Reliability bins per class (one-vs-rest)
        - Hosmer-Lemeshow per class (one-vs-rest)
    """
    K = detect_num_classes(df, num_classes)
    per_group: list[dict[str, Any]] = []
    excluded: list[str] = []
    brier_values: list[float] = []
    for grp, sub in df.groupby(attribute):
        if len(sub) < min_group_n:
            excluded.append(str(grp))
            continue
        try:
            score_mat = _score_matrix(sub, K)
        except ValueError as e:
            warnings.warn(
                f"Skipping group {grp}: cannot build score matrix ({e})",
                stacklevel=2,
            )
            continue
        y_true_g = sub["y_true"].to_numpy().astype(int)
        brier = brier_score(y_true_g, score_mat, K)
        per_class = per_class_brier(y_true_g, score_mat, K)
        per_class_reliability: list[dict[str, Any]] = []
        per_class_hl: list[dict[str, Any]] = []
        for k in range(K):
            y_k = _ovr_y_true(y_true_g, k)
            s_k = _binary_score_for_class(score_mat, k)
            per_class_reliability.append(
                {
                    "class_index": k,
                    **reliability_bins(y_k, s_k, n_bins=n_reliability_bins),
                }
            )
            per_class_hl.append(
                {
                    "class_index": k,
                    **hosmer_lemeshow(y_k, s_k, n_bins=n_hl_bins),
                }
            )
        per_group.append(
            {
                "group": str(grp),
                "n": len(sub),
                "brier_score": float(brier),
                "per_class_brier": [float(v) for v in per_class],
                "reliability_by_class": per_class_reliability,
                "hosmer_lemeshow_by_class": per_class_hl,
            }
        )
        if not np.isnan(brier):
            brier_values.append(float(brier))
    if excluded:
        warnings.warn(
            f"Excluding {len(excluded)} groups of '{attribute}' from calibration "
            f"(n < {min_group_n}): {excluded}",
            stacklevel=2,
        )
    brier_gap = (max(brier_values) - min(brier_values)) if len(brier_values) >= 2 else 0.0
    return {
        "attribute": attribute,
        "num_classes": K,
        "per_group": per_group,
        "excluded_groups": excluded,
        "brier_gap": float(brier_gap),
        "n_reliability_bins": n_reliability_bins,
        "n_hl_bins": n_hl_bins,
    }


def build_calibration_block(
    df: pd.DataFrame,
    protected_attrs: list[str],
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    n_reliability_bins: int = RELIABILITY_DEFAULT_BINS,
    n_hl_bins: int = HL_DEFAULT_BINS,
) -> dict[str, Any]:
    """Top-level orchestrator. Returns the evidence-pack ``calibration`` dict."""
    K = detect_num_classes(df, num_classes)
    score_mat = _score_matrix(df, K)
    y_true = df["y_true"].to_numpy().astype(int)
    global_brier = brier_score(y_true, score_mat, K)
    global_per_class = per_class_brier(y_true, score_mat, K)
    per_attribute = {
        a: per_group_calibration(
            df,
            a,
            num_classes=K,
            min_group_n=min_group_n,
            n_reliability_bins=n_reliability_bins,
            n_hl_bins=n_hl_bins,
        )
        for a in protected_attrs
    }
    return {
        "num_classes": K,
        "min_group_n": min_group_n,
        "n_reliability_bins": n_reliability_bins,
        "n_hl_bins": n_hl_bins,
        "global_brier_score": float(global_brier),
        "global_per_class_brier": [float(v) for v in global_per_class],
        "per_attribute": per_attribute,
    }
