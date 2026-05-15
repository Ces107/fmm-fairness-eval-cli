"""CLI entrypoint: `fmm-fairness evaluate ...`."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from fmm_fairness import __version__
from fmm_fairness.evidence import EvaluationConfig, write_evidence_pack

REQUIRED_COLUMNS = ["y_true", "y_pred", "y_score"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fmm-fairness",
        description="SaMD-specific fairness evaluation CLI for foundation-model medical AI.",
    )
    p.add_argument("--version", action="version", version=f"fmm-fairness-eval {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ev = sub.add_parser(
        "evaluate",
        help="Evaluate a predictions CSV and emit a fairness evidence pack.",
    )
    ev.add_argument("predictions", help="Path to predictions CSV (y_true,y_pred,y_score,<attrs>).")
    ev.add_argument(
        "--protected-attrs",
        required=True,
        help="Comma-separated protected attribute column names (e.g. 'site,sex,age_bucket').",
    )
    ev.add_argument(
        "--site-attribute",
        default="site",
        help="Column name to treat as the site/hospital identifier (default: site).",
    )
    ev.add_argument(
        "--output",
        default="fairness-report",
        help="Output directory for the evidence pack (default: fairness-report/).",
    )
    ev.add_argument(
        "--manifest-mode",
        choices=["ai-act"],
        default=None,
        help="Emit an additional regulatory mapping block. Currently supported: ai-act.",
    )
    return p


def _validate_dataframe(df: pd.DataFrame, protected_attrs: list[str]) -> list[str]:
    """Return list of error messages; empty = OK."""
    errors: list[str] = []
    missing_core = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_core:
        errors.append(f"Missing required columns: {missing_core}. "
                      f"Required: {REQUIRED_COLUMNS}.")
    missing_attrs = [a for a in protected_attrs if a not in df.columns]
    if missing_attrs:
        errors.append(f"Declared protected attributes not in CSV: {missing_attrs}.")
    if "y_true" in df.columns and not set(df["y_true"].unique()).issubset({0, 1}):
        errors.append("y_true must be in {0, 1}.")
    if "y_pred" in df.columns and not set(df["y_pred"].unique()).issubset({0, 1}):
        errors.append("y_pred must be in {0, 1}.")
    if "y_score" in df.columns:
        s = df["y_score"]
        if s.min() < 0.0 or s.max() > 1.0:
            errors.append("y_score must be in [0.0, 1.0].")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "evaluate":
        return 2
    try:
        df = pd.read_csv(args.predictions)
    except (OSError, pd.errors.ParserError) as e:
        print(f"ERROR: could not read predictions CSV: {e}", file=sys.stderr)
        return 1
    protected = [a.strip() for a in args.protected_attrs.split(",") if a.strip()]
    errs = _validate_dataframe(df, protected)
    if errs:
        for err in errs:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    cfg = EvaluationConfig(
        predictions_path=args.predictions,
        protected_attrs=protected,
        site_attribute=args.site_attribute,
        manifest_mode=args.manifest_mode,
        output_dir=args.output,
    )
    result = write_evidence_pack(df, cfg)
    print(f"OK: wrote evidence pack to {args.output}/")
    print(f"  - {result['report_md']}  (sha256={result['report_md_sha256'][:12]}...)")
    print(f"  - {result['evidence_json']}  (sha256={result['evidence_json_sha256'][:12]}...)")
    print(f"  - {result['audit_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
