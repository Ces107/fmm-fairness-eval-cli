"""Inter-rater agreement metrics for SaMD validation evidence.

The TFG that this CLI is named after evaluated a 6-class dermatopathology
model against 10 pathologists' annotations. The headline validation
evidence was the Cohen kappa matrix of pairwise agreement among the raters
plus the model, and the AI-vs-pooled-raters kappa with its bootstrap CI.

This module implements the canonical agreement statistics required to
reproduce that evidence:

- Pairwise Cohen kappa matrix (raters x raters, plus optional AI column).
- Fleiss kappa (multi-rater global agreement, fixed-N raters per item).
- Krippendorff alpha (multi-rater global agreement, tolerates missing
  ratings; nominal scale by default).
- AI-vs-pooled-raters kappa with percentile bootstrap CI.

Convention
----------

A "rater column" is an integer column in the input DataFrame. Each row is
one item (e.g. one whole-slide image); each rater column carries that
rater's class assignment. The reserved value ``missing_value`` (default
``-1``, matching the TFG convention) marks an unrated cell.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

MISSING_VALUE_DEFAULT = -1
BOOTSTRAP_ITERS_DEFAULT = 1000
RNG_SEED_DEFAULT = 42


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class AgreementResult:
    """One inter-rater statistic with optional bootstrap CI."""

    metric_name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n_items: int = 0
    raters: list[str] | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "metric": self.metric_name,
            "value": float(self.value) if not np.isnan(self.value) else None,
            "n_items": int(self.n_items),
        }
        if self.ci_low is not None and not np.isnan(self.ci_low):
            d["ci_low"] = float(self.ci_low)
        if self.ci_high is not None and not np.isnan(self.ci_high):
            d["ci_high"] = float(self.ci_high)
        if self.raters is not None:
            d["raters"] = list(self.raters)
        if self.note is not None:
            d["note"] = self.note
        return d


@dataclass
class KappaMatrixResult:
    """Pairwise Cohen kappa matrix as a labelled DataFrame plus serialisable form."""

    matrix: pd.DataFrame
    raters: list[str]
    stratum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        m = self.matrix.astype(float)
        # Replace NaN with None for clean JSON.
        rows = [
            [None if np.isnan(v) else float(v) for v in m.iloc[i].tolist()]
            for i in range(len(m))
        ]
        return {
            "raters": list(self.raters),
            "matrix": rows,
            "stratum": self.stratum,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drop_missing_pair(
    a: np.ndarray, b: np.ndarray, missing_value: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drop indices where either a or b carries the missing sentinel."""
    mask = (a != missing_value) & (b != missing_value)
    return a[mask], b[mask]


def _majority_vote(row: np.ndarray, missing_value: int) -> int:
    """Majority class per item; ties broken to the lowest class index. -1 if all missing."""
    valid = [int(r) for r in row if int(r) != missing_value]
    if not valid:
        return missing_value
    counts = Counter(valid)
    max_count = max(counts.values())
    return min(k for k, v in counts.items() if v == max_count)


def pool_raters(
    df: pd.DataFrame,
    rater_cols: list[str],
    *,
    missing_value: int = MISSING_VALUE_DEFAULT,
) -> np.ndarray:
    """Majority-vote rating per item; missing sentinel for items with no valid rater."""
    arr = df[rater_cols].to_numpy(dtype=int)
    return np.asarray(
        [_majority_vote(row, missing_value) for row in arr], dtype=int
    )


# ---------------------------------------------------------------------------
# Cohen kappa pairwise matrix
# ---------------------------------------------------------------------------


def cohen_kappa_matrix(
    df: pd.DataFrame,
    rater_cols: list[str],
    ai_col: str | None = None,
    *,
    missing_value: int = MISSING_VALUE_DEFAULT,
    stratify_by: str | None = None,
) -> dict[str, KappaMatrixResult] | KappaMatrixResult:
    """Pairwise Cohen kappa matrix of raters (optionally + AI column).

    Per-cell value is the Cohen kappa restricted to rows where both raters
    have a non-missing rating. Diagonal entries are 1.0 by construction.

    If ``stratify_by`` names a column, returns a dict of stratum-value to
    KappaMatrixResult. Otherwise returns a single KappaMatrixResult.
    """
    if stratify_by is None:
        return _kappa_matrix(df, rater_cols, ai_col, missing_value, stratum=None)
    out: dict[str, KappaMatrixResult] = {}
    for s, sub in df.groupby(stratify_by):
        out[str(s)] = _kappa_matrix(
            sub.reset_index(drop=True),
            rater_cols,
            ai_col,
            missing_value,
            stratum=str(s),
        )
    return out


def _kappa_matrix(
    df: pd.DataFrame,
    rater_cols: list[str],
    ai_col: str | None,
    missing_value: int,
    *,
    stratum: str | None,
) -> KappaMatrixResult:
    cols = list(rater_cols)
    if ai_col is not None:
        cols = [*cols, ai_col]
    raters_arr = {c: df[c].to_numpy(dtype=int) for c in cols}
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i in cols:
        for j in cols:
            if i == j:
                mat.loc[i, j] = 1.0
                continue
            a, b = _drop_missing_pair(raters_arr[i], raters_arr[j], missing_value)
            if len(a) < 2:
                mat.loc[i, j] = float("nan")
                continue
            try:
                mat.loc[i, j] = float(cohen_kappa_score(a, b))
            except ValueError:
                mat.loc[i, j] = float("nan")
    return KappaMatrixResult(matrix=mat, raters=cols, stratum=stratum)


# ---------------------------------------------------------------------------
# Fleiss kappa
# ---------------------------------------------------------------------------


def fleiss_kappa(
    df: pd.DataFrame,
    rater_cols: list[str],
    *,
    missing_value: int = MISSING_VALUE_DEFAULT,
) -> AgreementResult:
    """Fleiss' kappa for fixed-N raters per item.

    Items with any missing rating are excluded (Fleiss' kappa is defined for
    a fixed number of raters per item). For partially-missing-tolerant
    multi-rater agreement use ``krippendorff_alpha``.
    """
    matrix = df[rater_cols].to_numpy(dtype=int)
    valid_mask = (matrix != missing_value).all(axis=1)
    matrix = matrix[valid_mask]
    n_items, n_raters = matrix.shape
    if n_items == 0 or n_raters < 2:
        return AgreementResult(
            metric_name="fleiss_kappa",
            value=float("nan"),
            n_items=n_items,
            raters=list(rater_cols),
            note="Fleiss kappa undefined: no items with all raters present, or < 2 raters.",
        )
    categories = np.unique(matrix)
    n_categories = len(categories)
    # n_ij = count of raters who assigned item i to category j
    n_ij = np.zeros((n_items, n_categories), dtype=int)
    for j, c in enumerate(categories):
        n_ij[:, j] = (matrix == c).sum(axis=1)
    # Fleiss' formula
    p_i = (np.sum(n_ij ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    p_j = n_ij.sum(axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j ** 2))
    if 1.0 - p_e == 0.0:
        return AgreementResult(
            metric_name="fleiss_kappa",
            value=float("nan"),
            n_items=n_items,
            raters=list(rater_cols),
            note="Fleiss kappa undefined: expected agreement is 1.",
        )
    kappa = (p_bar - p_e) / (1.0 - p_e)
    return AgreementResult(
        metric_name="fleiss_kappa",
        value=float(kappa),
        n_items=n_items,
        raters=list(rater_cols),
    )


# ---------------------------------------------------------------------------
# Krippendorff alpha (nominal)
# ---------------------------------------------------------------------------


def krippendorff_alpha(
    df: pd.DataFrame,
    rater_cols: list[str],
    *,
    missing_value: int = MISSING_VALUE_DEFAULT,
) -> AgreementResult:
    """Krippendorff's alpha (nominal scale) tolerating missing ratings.

    Implementation follows Krippendorff (2011): the difference function on
    nominal categories is ``delta(a, b) = 1 if a != b else 0``; the observed
    disagreement is computed per item over the actually-rated pairs, the
    expected disagreement is computed from the marginal frequencies of
    valid ratings.
    """
    raw = np.array(df[rater_cols].to_numpy(dtype=float), copy=True)
    # Sentinel -> NaN, so vectorised pair-counting can skip missing cells.
    raw[raw == float(missing_value)] = np.nan
    n_items = raw.shape[0]
    if n_items == 0:
        return AgreementResult(
            metric_name="krippendorff_alpha",
            value=float("nan"),
            n_items=0,
            raters=list(rater_cols),
        )

    # Observed disagreement.
    pairs_total = 0
    disagree_total = 0
    for i in range(n_items):
        row = raw[i, :]
        valid = row[~np.isnan(row)]
        m_i = len(valid)
        if m_i < 2:
            continue
        # number of ordered pairs (j, k) with j != k
        pairs_total += m_i * (m_i - 1)
        # disagreements: 2 * unordered disagreements (since ordered)
        # use Counter on integer cast to count category multiplicities
        counter = Counter(int(v) for v in valid)
        equal_pairs = sum(c * (c - 1) for c in counter.values())
        disagree_total += m_i * (m_i - 1) - equal_pairs

    if pairs_total == 0:
        return AgreementResult(
            metric_name="krippendorff_alpha",
            value=float("nan"),
            n_items=n_items,
            raters=list(rater_cols),
            note="Alpha undefined: fewer than 2 valid ratings per item.",
        )
    do = disagree_total / pairs_total

    # Expected disagreement from marginal category frequencies.
    flat = raw[~np.isnan(raw)].astype(int)
    _, counts = np.unique(flat, return_counts=True)
    n_total = int(counts.sum())
    if n_total < 2:
        return AgreementResult(
            metric_name="krippendorff_alpha",
            value=float("nan"),
            n_items=n_items,
            raters=list(rater_cols),
        )
    # de = sum over c != d of (n_c * n_d) / (n_total * (n_total - 1))
    # equivalently 1 - sum(n_c * (n_c - 1)) / (n_total * (n_total - 1))
    equal_total = int(np.sum(counts * (counts - 1)))
    de_numerator = n_total * (n_total - 1) - equal_total
    de_denominator = n_total * (n_total - 1)
    de = de_numerator / de_denominator
    if de == 0:
        return AgreementResult(
            metric_name="krippendorff_alpha",
            value=float("nan"),
            n_items=n_items,
            raters=list(rater_cols),
            note="Alpha undefined: expected disagreement is 0 (single category).",
        )
    alpha = 1.0 - (do / de)
    return AgreementResult(
        metric_name="krippendorff_alpha",
        value=float(alpha),
        n_items=n_items,
        raters=list(rater_cols),
    )


# ---------------------------------------------------------------------------
# AI vs pooled-raters Cohen kappa with bootstrap CI
# ---------------------------------------------------------------------------


def ai_vs_pooled_raters_kappa(
    df: pd.DataFrame,
    rater_cols: list[str],
    ai_col: str,
    *,
    missing_value: int = MISSING_VALUE_DEFAULT,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    seed: int = RNG_SEED_DEFAULT,
) -> AgreementResult:
    """Cohen kappa between the AI prediction and the per-item majority of the raters.

    Items where every rater is missing are excluded; items where the AI is
    missing are also excluded (the AI is expected to produce a prediction).
    Bootstrap CI is percentile, resampled at the item level.
    """
    pooled = pool_raters(df, rater_cols, missing_value=missing_value)
    ai = df[ai_col].to_numpy(dtype=int)
    valid_mask = (pooled != missing_value) & (ai != missing_value)
    pooled_v = pooled[valid_mask]
    ai_v = ai[valid_mask]
    if len(pooled_v) < 2:
        return AgreementResult(
            metric_name="ai_vs_pooled_raters_kappa",
            value=float("nan"),
            n_items=int(valid_mask.sum()),
            raters=list(rater_cols),
            note="AI-vs-pooled kappa undefined: fewer than 2 items with both AI and a valid pooled rating.",
        )
    point = float(cohen_kappa_score(pooled_v, ai_v))

    rng = np.random.default_rng(seed)
    n = len(pooled_v)
    ks: list[float] = []
    for _ in range(bootstrap_iters):
        idx = rng.integers(0, n, size=n)
        ks.append(float(cohen_kappa_score(pooled_v[idx], ai_v[idx])))
    ci_low = float(np.percentile(ks, 2.5)) if ks else float("nan")
    ci_high = float(np.percentile(ks, 97.5)) if ks else float("nan")
    return AgreementResult(
        metric_name="ai_vs_pooled_raters_kappa",
        value=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_items=int(n),
        raters=list(rater_cols),
    )


# ---------------------------------------------------------------------------
# High-level evidence builder
# ---------------------------------------------------------------------------


def build_inter_rater_evidence(
    df: pd.DataFrame,
    rater_cols: list[str],
    ai_col: str | None = None,
    *,
    missing_value: int = MISSING_VALUE_DEFAULT,
    stratify_by: str | None = None,
    bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    seed: int = RNG_SEED_DEFAULT,
) -> dict[str, Any]:
    """Assemble the inter-rater agreement block for the evidence pack."""
    block: dict[str, Any] = {
        "rater_columns": list(rater_cols),
        "ai_column": ai_col,
        "missing_value_sentinel": missing_value,
        "fleiss_kappa": fleiss_kappa(
            df, rater_cols, missing_value=missing_value
        ).to_dict(),
        "krippendorff_alpha": krippendorff_alpha(
            df, rater_cols, missing_value=missing_value
        ).to_dict(),
    }
    matrix = cohen_kappa_matrix(
        df, rater_cols, ai_col, missing_value=missing_value
    )
    assert isinstance(matrix, KappaMatrixResult)
    block["cohen_kappa_matrix"] = matrix.to_dict()
    if ai_col is not None:
        block["ai_vs_pooled_raters_kappa"] = ai_vs_pooled_raters_kappa(
            df,
            rater_cols,
            ai_col,
            missing_value=missing_value,
            bootstrap_iters=bootstrap_iters,
            seed=seed,
        ).to_dict()
    if stratify_by is not None and stratify_by in df.columns:
        stratified = cohen_kappa_matrix(
            df,
            rater_cols,
            ai_col,
            missing_value=missing_value,
            stratify_by=stratify_by,
        )
        assert isinstance(stratified, dict)
        block["cohen_kappa_matrix_by_stratum"] = {
            k: v.to_dict() for k, v in stratified.items()
        }
        block["stratified_by"] = stratify_by
    return block
