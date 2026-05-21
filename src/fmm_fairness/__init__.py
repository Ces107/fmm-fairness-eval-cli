"""fmm-fairness-eval — SaMD-specific fairness evaluation for medical AI.

Public API entrypoints:
- metrics.* — per-attribute fairness metrics (binary and multi-class F1 family)
- evidence.* — emit Markdown report + JSON evidence pack with SHA-256 chain
- cli.main — argparse CLI entrypoint
"""

__version__ = "0.2.0a1"

from fmm_fairness import evidence, metrics

__all__ = ["__version__", "evidence", "metrics"]
