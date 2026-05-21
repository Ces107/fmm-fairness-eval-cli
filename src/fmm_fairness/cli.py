"""CLI entrypoint: ``fmm-fairness evaluate ...``."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from fmm_fairness import __version__
from fmm_fairness.evidence import EvaluationConfig, write_evidence_pack
from fmm_fairness.metrics import (
    SCORE_COL_BINARY,
    SCORE_COL_PREFIX,
    _multi_class_score_columns,
    detect_num_classes,
)

LABEL_COLUMNS = ["y_true", "y_pred"]


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
    ev.add_argument(
        "predictions",
        help=(
            "Path to predictions CSV. Required columns: y_true, y_pred, plus a "
            "score column shape: binary 'y_score' (single col in [0,1]) or "
            "multi-class 'y_score_0..y_score_{K-1}' (per-class probabilities)."
        ),
    )
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
        "--num-classes",
        type=int,
        default=None,
        help=(
            "Number of classes K (>= 2). Optional; auto-detected from the score "
            "columns when omitted. A mismatch with the detected shape triggers a warning."
        ),
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


def _validate_dataframe(
    df: pd.DataFrame, protected_attrs: list[str], num_classes_arg: int | None
) -> tuple[list[str], int | None]:
    """Return (error messages, resolved K). Empty error list => OK."""
    errors: list[str] = []

    missing_labels = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing_labels:
        errors.append(
            f"Missing required label columns: {missing_labels}. "
            f"Required: {LABEL_COLUMNS}.",
        )

    missing_attrs = [a for a in protected_attrs if a not in df.columns]
    if missing_attrs:
        errors.append(f"Declared protected attributes not in CSV: {missing_attrs}.")

    multi_cols = _multi_class_score_columns(df) if not missing_labels else []
    has_binary = SCORE_COL_BINARY in df.columns
    if not multi_cols and not has_binary:
        errors.append(
            f"No score columns found. Provide '{SCORE_COL_BINARY}' (binary) "
            f"or '{SCORE_COL_PREFIX}0..{SCORE_COL_PREFIX}K-1' (multi-class).",
        )

    if errors:
        return errors, None

    try:
        K = detect_num_classes(df, num_classes_arg)
    except ValueError as e:
        return [str(e)], None

    label_max_true = int(df["y_true"].max())
    label_max_pred = int(df["y_pred"].max())
    label_min_true = int(df["y_true"].min())
    label_min_pred = int(df["y_pred"].min())
    if label_min_true < 0 or label_min_pred < 0:
        errors.append("y_true and y_pred must be non-negative integers.")
    if label_max_true >= K or label_max_pred >= K:
        errors.append(
            f"y_true / y_pred contain class indices outside [0, {K - 1}] "
            f"(max y_true={label_max_true}, max y_pred={label_max_pred}, K={K}). "
            f"Either widen --num-classes or fix the predictions CSV.",
        )

    if K == 2 and SCORE_COL_BINARY in df.columns and not multi_cols:
        s = df[SCORE_COL_BINARY]
        if s.min() < 0.0 or s.max() > 1.0:
            errors.append(f"{SCORE_COL_BINARY} must be in [0.0, 1.0].")
    if multi_cols:
        score_block = df[multi_cols]
        if (score_block.values < 0.0).any() or (score_block.values > 1.0).any():
            errors.append(
                f"Multi-class score columns must be in [0.0, 1.0]; columns: {multi_cols}.",
            )

    return errors, K


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
    errs, resolved_k = _validate_dataframe(df, protected, args.num_classes)
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
        num_classes=resolved_k,
    )
    result = write_evidence_pack(df, cfg)
    print(f"OK: wrote evidence pack to {args.output}/")
    print(f"  K (number of classes) = {resolved_k}")
    print(f"  - {result['report_md']}  (sha256={result['report_md_sha256'][:12]}...)")
    print(f"  - {result['evidence_json']}  (sha256={result['evidence_json_sha256'][:12]}...)")
    print(f"  - {result['audit_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
