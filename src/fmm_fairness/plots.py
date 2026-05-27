"""Optional reliability-diagram rendering.

Calling ``render_reliability_plots(...)`` paints one PNG per (attribute, group,
class) reliability diagram. The numeric calibration data lives in
``calibration.reliability_bins`` regardless of whether this module produces
plots; the plots are a cosmetic supplement aimed at regulator-facing reports.

Matplotlib is an optional dependency. When it is not installed, this module
warns and returns an empty list of paths; the CLI continues to emit JSON +
Markdown + audit hash without plots.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fmm_fairness.calibration import reliability_bins
from fmm_fairness.metrics import (
    MIN_GROUP_N_DEFAULT,
    _score_matrix,
    detect_num_classes,
)


def _try_import_matplotlib() -> Any:
    """Lazy import so the package never hard-fails on systems without matplotlib.

    Avoid clobbering a caller's existing backend (TD-006). Only force the
    headless ``Agg`` backend when matplotlib has not yet selected one of its
    own. A notebook session that previously imported matplotlib with the
    ``inline`` backend keeps it; a fresh process gets ``Agg``.
    """
    try:
        import matplotlib

        current = matplotlib.get_backend()
        # Matplotlib's default before any explicit `use()` call is ``agg``;
        # any GUI / inline backend will be a non-default string.
        if current.lower() in {"", "agg"}:
            matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _slug(value: str) -> str:
    """Best-effort filesystem-friendly slug for plot filenames."""
    out: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "unnamed"


def render_reliability_plots(
    df: pd.DataFrame,
    protected_attrs: list[str],
    output_dir: Path,
    *,
    num_classes: int | None = None,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
    n_bins: int = 10,
) -> list[str]:
    """Render one reliability PNG per (attribute, group, class).

    Returns a list of relative file paths written, in stable order. Returns
    ``[]`` (and warns once) if matplotlib is not installed.

    The diagram is the standard accuracy-vs-confidence plot with the
    ``y = x`` perfect-calibration reference line. Each plot is self-contained;
    no shared figure state is leaked between calls.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        warnings.warn(
            "matplotlib not installed; --render-plots is a no-op. "
            "Install the [plots] optional dependency to enable PNG output.",
            stacklevel=2,
        )
        return []
    K = detect_num_classes(df, num_classes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for attr in protected_attrs:
        if attr not in df.columns:
            continue
        for grp_name, sub in df.groupby(attr):
            if len(sub) < min_group_n:
                continue
            try:
                score_mat = _score_matrix(sub, K)
            except ValueError:
                continue
            y_true_g = sub["y_true"].to_numpy().astype(int)
            for k in range(K):
                y_k = (y_true_g == k).astype(int)
                s_k = score_mat[:, k]
                bins = reliability_bins(y_k, s_k, n_bins=n_bins)
                fig, ax = plt.subplots(figsize=(4.0, 4.0))
                centers = np.array(bins["bin_centers"], dtype=float)
                accs = np.array(bins["accuracies"], dtype=float)
                confs = np.array(bins["confidences"], dtype=float)
                counts = np.array(bins["counts"], dtype=int)
                ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0)
                width = 1.0 / max(n_bins, 1)
                valid = ~np.isnan(accs)
                ax.bar(
                    centers[valid] - width / 2,
                    accs[valid],
                    width=width * 0.85,
                    color="#1f77b4",
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )
                # Confidence marks as small ticks per bin.
                for c, conf in zip(centers[valid], confs[valid], strict=True):
                    ax.plot([c, c], [conf, conf], marker="o", color="black", markersize=3)
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_xlabel("Predicted probability (confidence)")
                ax.set_ylabel("Observed positive rate (accuracy)")
                ax.set_title(
                    f"Reliability: {attr}={grp_name}, class={k}\n"
                    f"n={int(counts.sum())}"
                )
                ax.grid(True, alpha=0.2)
                rel_path = (
                    f"reliability_{_slug(attr)}_{_slug(str(grp_name))}_c{k}.png"
                )
                full_path = output_dir / rel_path
                fig.tight_layout()
                fig.savefig(full_path, dpi=120)
                plt.close(fig)
                paths.append(str(full_path))
    return paths
