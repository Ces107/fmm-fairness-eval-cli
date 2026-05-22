"""fmm-fairness-eval — SaMD-specific fairness evaluation for medical AI.

Public API entrypoints:
- metrics.* — per-attribute fairness metrics (binary and multi-class F1 family)
- evidence.* — emit Markdown report + JSON evidence pack with SHA-256 chain
- cli.main — argparse CLI entrypoint
"""

__version__ = "0.2.0a4"

from fmm_fairness import agreement, comparison, evidence, metrics, statistics

__all__ = [
    "__version__",
    "agreement",
    "comparison",
    "evidence",
    "metrics",
    "statistics",
]
