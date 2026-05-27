"""Statistical-rigour primitives: BCa bootstrap, permutation test, MDE.

This module turns the v0.1 percentile-bootstrap CI into a defensible
research-grade inference layer:

- ``bca_bootstrap_gap_ci(...)`` returns a bias-corrected and accelerated
  (Efron 1987) bootstrap CI over a per-group statistic's max-min gap.
  Falls back to the percentile CI when BCa is undefined (e.g. degenerate
  jackknife with a constant statistic).
- ``permutation_test_gap_pvalue(...)`` runs a label-shuffle permutation
  test for H0 = "no gap"; returns a two-sided p-value.
- ``minimum_detectable_effect(...)`` reports the gap size that the
  current sample would detect at alpha=0.05, power=0.80, given the
  bootstrap-estimated standard error of the gap statistic.
- ``cohens_d(...)`` and ``odds_ratio_binary(...)`` provide standard
  effect-size summaries between two groups.

References
----------

- Efron, B. (1987). *Better Bootstrap Confidence Intervals.*
  Journal of the American Statistical Association 82(397):171-185.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap.*
  Chapman & Hall, chapter 14.
- DiCiccio, T. J., Efron, B. (1996). *Bootstrap Confidence Intervals.*
  Statistical Science 11(3):189-228.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

PERMUTATION_ITERS_DEFAULT = 1000
BOOTSTRAP_ITERS_DEFAULT = 1000
ALPHA_DEFAULT = 0.05
POWER_DEFAULT = 0.80


@dataclass
class GapInference:
    """Statistical-rigour bundle for one gap statistic."""

    gap: float
    bootstrap_method: str        # "bca" | "percentile"
    ci_low: float
    ci_high: float
    bootstrap_se: float
    permutation_p_value: float | None = None
    permutation_iters: int | None = None
    minimum_detectable_effect: float | None = None
    alpha: float = ALPHA_DEFAULT
    power: float = POWER_DEFAULT

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap": float(self.gap),
            "bootstrap_method": self.bootstrap_method,
            "ci_low": float(self.ci_low) if not np.isnan(self.ci_low) else None,
            "ci_high": float(self.ci_high) if not np.isnan(self.ci_high) else None,
            "bootstrap_se": (
                float(self.bootstrap_se) if not np.isnan(self.bootstrap_se) else None
            ),
            "permutation_p_value": (
                float(self.permutation_p_value)
                if self.permutation_p_value is not None
                else None
            ),
            "permutation_iters": self.permutation_iters,
            "minimum_detectable_effect": (
                float(self.minimum_detectable_effect)
                if self.minimum_detectable_effect is not None
                else None
            ),
            "alpha": float(self.alpha),
            "power": float(self.power),
        }


# ---------------------------------------------------------------------------
# Gap statistic helper
# ---------------------------------------------------------------------------


def _gap_statistic(
    df: pd.DataFrame, attribute: str, per_group_fn: Callable[[pd.DataFrame], float]
) -> float:
    """Apply ``per_group_fn`` per group and return max-min."""
    vals: list[float] = []
    for _, sub in df.groupby(attribute):
        v = per_group_fn(sub)
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 2:
        return 0.0
    return float(max(vals) - min(vals))


# ---------------------------------------------------------------------------
# BCa bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_replicates(
    df: pd.DataFrame,
    attribute: str,
    per_group_fn: Callable[[pd.DataFrame], float],
    n_iters: int,
    seed: int,
) -> np.ndarray:
    """Stratified bootstrap draws of the gap statistic."""
    rng = np.random.default_rng(seed)
    groups = df[attribute].unique()
    indices_by_group = {g: df.index[df[attribute] == g].to_numpy() for g in groups}
    out = np.empty(n_iters, dtype=float)
    for i in range(n_iters):
        vals = []
        for g in groups:
            idx = indices_by_group[g]
            sample_idx = rng.choice(idx, size=len(idx), replace=True)
            sample = df.loc[sample_idx]
            v = per_group_fn(sample)
            if not np.isnan(v):
                vals.append(v)
        out[i] = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
    return out


def _jackknife_gap(
    df: pd.DataFrame,
    attribute: str,
    per_group_fn: Callable[[pd.DataFrame], float],
) -> np.ndarray:
    """Leave-one-out jackknife replicates of the gap statistic."""
    n = len(df)
    out = np.empty(n, dtype=float)
    df_arr = df.reset_index(drop=True)
    for i in range(n):
        out[i] = _gap_statistic(df_arr.drop(index=i), attribute, per_group_fn)
    return out


def _bca_endpoints(
    theta_hat: float,
    theta_boot: np.ndarray,
    theta_jack: np.ndarray,
    alpha: float,
) -> tuple[float, float, float, float]:
    """Return (z0, a, alpha1, alpha2) for the BCa CI.

    Falls back: returns (0.0, 0.0, alpha/2, 1 - alpha/2) when the
    acceleration computation is undefined (zero-variance jackknife) or
    when the bias correction is infinite (all bootstrap reps equal).
    """
    below = float(np.mean(theta_boot < theta_hat))
    if below in (0.0, 1.0):
        z0 = 0.0
    else:
        z0 = float(norm.ppf(below))
    theta_dot = float(theta_jack.mean())
    diffs = theta_dot - theta_jack
    num = float(np.sum(diffs ** 3))
    den = float(6.0 * (np.sum(diffs ** 2) ** 1.5))
    a = num / den if den != 0.0 else 0.0
    z_lo = float(norm.ppf(alpha / 2))
    z_hi = float(norm.ppf(1.0 - alpha / 2))
    def _adj(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if denom == 0.0:
            return float("nan")
        return float(norm.cdf(z0 + (z0 + z) / denom))
    alpha1 = _adj(z_lo)
    alpha2 = _adj(z_hi)
    if np.isnan(alpha1) or np.isnan(alpha2) or alpha1 >= alpha2:
        return 0.0, 0.0, alpha / 2.0, 1.0 - alpha / 2.0
    return z0, a, alpha1, alpha2


def _coerce_bracket(
    theta_hat: float, ci_low: float, ci_high: float
) -> tuple[float, float]:
    """Widen a bootstrap CI so it contains the point estimate.

    The max-min gap statistic is *positively biased*: under resampling, the
    group maximum drifts up and the minimum drifts down, so for a true gap
    near its 0 boundary the bootstrap distribution sits above ``theta_hat``
    and the BCa (or percentile) interval can EXCLUDE the point estimate
    (e.g. theta_hat=0.0073 with CI [0.0124, 0.1259]). An interval that does
    not contain its own estimator is not interpretable as a confidence
    interval for that estimator.

    We resolve this by widening the interval to include ``theta_hat``. This
    is strictly conservative — it never narrows the interval, so coverage is
    not reduced. The bootstrap SE and the permutation p-value continue to
    carry the real uncertainty signal (a gap that triggers this coercion is
    typically not distinguishable from zero, which the permutation p-value
    will reflect). This keeps reported intervals coherent without hiding the
    near-zero, high-bias regime.
    """
    return min(ci_low, theta_hat), max(ci_high, theta_hat)


def bca_bootstrap_gap_ci(
    df: pd.DataFrame,
    attribute: str,
    per_group_fn: Callable[[pd.DataFrame], float],
    *,
    n_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    seed: int = 42,
) -> tuple[float, float, float, np.ndarray]:
    """Return (ci_low, ci_high, bootstrap_se, theta_boot) using BCa.

    Falls back to the percentile interval when BCa endpoints are degenerate.
    """
    theta_hat = _gap_statistic(df, attribute, per_group_fn)
    theta_boot = _bootstrap_replicates(df, attribute, per_group_fn, n_iters, seed)
    theta_jack = _jackknife_gap(df, attribute, per_group_fn)
    _, _, alpha1, alpha2 = _bca_endpoints(theta_hat, theta_boot, theta_jack, alpha)
    ci_low = float(np.quantile(theta_boot, alpha1))
    ci_high = float(np.quantile(theta_boot, alpha2))
    ci_low, ci_high = _coerce_bracket(theta_hat, ci_low, ci_high)
    se = float(np.std(theta_boot, ddof=1))
    return ci_low, ci_high, se, theta_boot


def percentile_bootstrap_gap_ci(
    df: pd.DataFrame,
    attribute: str,
    per_group_fn: Callable[[pd.DataFrame], float],
    *,
    n_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    seed: int = 42,
) -> tuple[float, float, float, np.ndarray]:
    """Plain percentile bootstrap CI; preserved for backward compatibility."""
    theta_hat = _gap_statistic(df, attribute, per_group_fn)
    theta_boot = _bootstrap_replicates(df, attribute, per_group_fn, n_iters, seed)
    ci_low = float(np.quantile(theta_boot, alpha / 2))
    ci_high = float(np.quantile(theta_boot, 1.0 - alpha / 2))
    ci_low, ci_high = _coerce_bracket(theta_hat, ci_low, ci_high)
    se = float(np.std(theta_boot, ddof=1))
    return ci_low, ci_high, se, theta_boot


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def permutation_test_gap_pvalue(
    df: pd.DataFrame,
    attribute: str,
    per_group_fn: Callable[[pd.DataFrame], float],
    *,
    n_iters: int = PERMUTATION_ITERS_DEFAULT,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Two-sided permutation test for H0 = "no gap".

    Procedure: shuffle the group labels uniformly at random, recompute the
    max-min gap, and count permutations whose statistic is at least as
    extreme as the observed. The p-value is the proportion of such
    permutations (with the +1 / +1 plug-in to avoid p=0).
    """
    rng = np.random.default_rng(seed)
    theta_hat = _gap_statistic(df, attribute, per_group_fn)
    labels = df[attribute].to_numpy()
    perm_stats = np.empty(n_iters, dtype=float)
    df_work = df.copy()
    for i in range(n_iters):
        shuffled = rng.permutation(labels)
        df_work[attribute] = shuffled
        perm_stats[i] = _gap_statistic(df_work, attribute, per_group_fn)
    n_extreme = int((perm_stats >= theta_hat).sum())
    p_value = (n_extreme + 1) / (n_iters + 1)
    return float(p_value), perm_stats


# ---------------------------------------------------------------------------
# Minimum detectable effect (post-hoc-friendly)
# ---------------------------------------------------------------------------


def minimum_detectable_effect(
    bootstrap_se: float,
    *,
    alpha: float = ALPHA_DEFAULT,
    power: float = POWER_DEFAULT,
) -> float:
    """Return the MDE = (z_{1-alpha/2} + z_{power}) * bootstrap_se.

    This is the standard sample-size formula adapted to the bootstrap
    estimate of the gap statistic's SE. It answers the operator's
    pre-experiment question: "given the current sample, how big a gap
    would I be able to detect at alpha=0.05, power=0.80?".
    """
    if np.isnan(bootstrap_se) or bootstrap_se <= 0.0:
        return float("nan")
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    return float((z_alpha + z_beta) * bootstrap_se)


# ---------------------------------------------------------------------------
# Effect sizes (between two groups)
# ---------------------------------------------------------------------------


def cohens_d(
    values_a: np.ndarray, values_b: np.ndarray
) -> float:
    """Standardised mean difference between two groups (pooled SD)."""
    n_a, n_b = len(values_a), len(values_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    mean_diff = float(np.mean(values_a) - np.mean(values_b))
    pooled_var = (
        (n_a - 1) * float(np.var(values_a, ddof=1))
        + (n_b - 1) * float(np.var(values_b, ddof=1))
    ) / (n_a + n_b - 2)
    if pooled_var <= 0.0:
        return float("nan")
    return mean_diff / sqrt(pooled_var)


def odds_ratio_binary(
    success_a: int, total_a: int, success_b: int, total_b: int
) -> float:
    """Odds ratio between two binary outcomes. Returns NaN if degenerate."""
    if total_a == 0 or total_b == 0:
        return float("nan")
    failure_a = total_a - success_a
    failure_b = total_b - success_b
    if failure_a == 0 or failure_b == 0 or success_b == 0:
        return float("nan")
    odds_a = success_a / failure_a if failure_a > 0 else float("inf")
    odds_b = success_b / failure_b if failure_b > 0 else float("inf")
    if odds_b == 0:
        return float("nan")
    return float(odds_a / odds_b)


# ---------------------------------------------------------------------------
# Top-level inference orchestrator
# ---------------------------------------------------------------------------


def gap_inference(
    df: pd.DataFrame,
    attribute: str,
    per_group_fn: Callable[[pd.DataFrame], float],
    *,
    bootstrap_method: str = "bca",
    n_bootstrap_iters: int = BOOTSTRAP_ITERS_DEFAULT,
    n_permutation_iters: int = PERMUTATION_ITERS_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    power: float = POWER_DEFAULT,
    seed: int = 42,
) -> GapInference:
    """One-stop inference bundle: CI + p-value + MDE for a per-group gap."""
    theta_hat = _gap_statistic(df, attribute, per_group_fn)
    if bootstrap_method == "bca":
        ci_low, ci_high, se, _ = bca_bootstrap_gap_ci(
            df, attribute, per_group_fn,
            n_iters=n_bootstrap_iters, alpha=alpha, seed=seed,
        )
    elif bootstrap_method == "percentile":
        ci_low, ci_high, se, _ = percentile_bootstrap_gap_ci(
            df, attribute, per_group_fn,
            n_iters=n_bootstrap_iters, alpha=alpha, seed=seed,
        )
    else:
        raise ValueError(
            f"bootstrap_method must be 'bca' or 'percentile'; got {bootstrap_method!r}.",
        )
    if n_permutation_iters > 0:
        p_value, _ = permutation_test_gap_pvalue(
            df, attribute, per_group_fn,
            n_iters=n_permutation_iters, seed=seed + 1,
        )
    else:
        p_value = None
    mde = minimum_detectable_effect(se, alpha=alpha, power=power)
    return GapInference(
        gap=theta_hat,
        bootstrap_method=bootstrap_method,
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_se=se,
        permutation_p_value=p_value,
        permutation_iters=n_permutation_iters if n_permutation_iters > 0 else None,
        minimum_detectable_effect=mde,
        alpha=alpha,
        power=power,
    )
