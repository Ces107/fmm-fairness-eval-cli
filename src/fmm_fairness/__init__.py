"""fmm-fairness-eval — SaMD-specific fairness evaluation for medical AI.

Public API entrypoints:
- metrics.* — per-attribute fairness metrics (binary and multi-class F1 family)
- agreement.* — inter-rater agreement statistics (Cohen, Fleiss, Krippendorff)
- statistics.* — BCa bootstrap, permutation test, MDE / Cohen's d / odds ratio
- comparison.* — Pareto-frontier foundation-model comparison
- intersect.* — intersectional cross-product fairness
- calibration.* — Brier score, Hosmer-Lemeshow, reliability bins
- plots.* — optional matplotlib reliability diagrams
- ai_act_dossier.* — EU AI Act Art. 9/10/13/14/15/72 dossier builder + templates
- evidence.* — emit Markdown report + JSON evidence pack with SHA-256 chain
- cli.main — argparse CLI entrypoint
"""

__version__ = "0.2.0a11"

from fmm_fairness import (
    agreement,
    ai_act_dossier,
    calibration,
    comparison,
    evidence,
    intersect,
    metrics,
    plots,
    statistics,
)

__all__ = [
    "__version__",
    "agreement",
    "ai_act_dossier",
    "calibration",
    "comparison",
    "evidence",
    "intersect",
    "metrics",
    "plots",
    "statistics",
]
