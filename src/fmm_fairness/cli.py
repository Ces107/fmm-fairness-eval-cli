"""CLI entrypoint: ``fmm-fairness evaluate ...`` and ``fmm-fairness compare ...``."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from fmm_fairness import __version__
from fmm_fairness.comparison import (
    compare_models,
    render_comparison_markdown,
)
from fmm_fairness.evidence import EvaluationConfig, write_evidence_pack
from fmm_fairness.metrics import (
    SCORE_COL_BINARY,
    SCORE_COL_PREFIX,
    _multi_class_score_columns,
    detect_num_classes,
)

LABEL_COLUMNS = ["y_true", "y_pred"]


def _add_evaluate(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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
        "--rater-cols",
        default=None,
        help=(
            "Comma-separated names of clinician-rater columns to use for inter-rater "
            "agreement (e.g. 'doc1,doc2,doc3,...,doc10')."
        ),
    )
    ev.add_argument(
        "--rater-missing-value",
        type=int,
        default=-1,
        help=("Sentinel value used in rater columns for an unrated cell (default: -1)."),
    )
    ev.add_argument(
        "--bootstrap-method",
        choices=["bca", "percentile"],
        default="bca",
        help=(
            "Bootstrap CI method for the F1 family gap metrics. 'bca' "
            "(default) is bias-corrected and accelerated (Efron 1987); "
            "'percentile' is the v0.1-compatible interval."
        ),
    )
    ev.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for the F1 family CIs (default: 1000).",
    )
    ev.add_argument(
        "--permutation-iters",
        type=int,
        default=0,
        help=(
            "If > 0, run a label-shuffle permutation test for H0='no gap' "
            "on the F1 family metrics and emit a p-value. Default 0 (off)."
        ),
    )
    ev.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the CI and MDE (default: 0.05).",
    )
    ev.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Power target for the minimum detectable effect (default: 0.80).",
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


def _add_compare(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cmp_ = sub.add_parser(
        "compare",
        help=(
            "Compare 2+ foundation-model candidates on a joint accuracy + fairness "
            "Pareto frontier and recommend one."
        ),
    )
    cmp_.add_argument(
        "predictions",
        nargs="+",
        help=(
            "Path to a predictions CSV per candidate. Same shape as `evaluate` "
            "(y_true, y_pred, y_score* columns, plus protected-attribute columns)."
        ),
    )
    cmp_.add_argument(
        "--labels",
        required=True,
        help=(
            "Comma-separated candidate labels, in the same order as the predictions "
            "paths (e.g. 'uni,conch,plip,transpath,gigapath,titan')."
        ),
    )
    cmp_.add_argument(
        "--protected-attrs",
        required=True,
        help="Comma-separated protected attribute column names (e.g. 'site,sex').",
    )
    cmp_.add_argument(
        "--site-attribute",
        default="site",
        help="Column name to treat as the site identifier (default: site).",
    )
    cmp_.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes K (>= 2). Auto-detected when omitted.",
    )
    cmp_.add_argument(
        "--fairness-floor",
        type=float,
        default=0.10,
        help=(
            "Maximum acceptable inter-site weighted-F1 gap for the Art. 9 "
            "recommendation (default: 0.10). Frontier candidates above this floor "
            "are flagged in the rationale."
        ),
    )
    cmp_.add_argument(
        "--output",
        default="comparison-report",
        help="Output directory for the comparison evidence pack.",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fmm-fairness",
        description="SaMD-specific fairness evaluation CLI for foundation-model medical AI.",
    )
    p.add_argument("--version", action="version", version=f"fmm-fairness-eval {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    _add_evaluate(sub)
    _add_compare(sub)
    return p


def _validate_dataframe(
    df: pd.DataFrame,
    protected_attrs: list[str],
    num_classes_arg: int | None,
    rater_cols: list[str] | None = None,
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

    if rater_cols:
        missing_rater_cols = [c for c in rater_cols if c not in df.columns]
        if missing_rater_cols:
            errors.append(
                f"Declared rater columns not in CSV: {missing_rater_cols}.",
            )

    return errors, K


def _run_evaluate(args: argparse.Namespace) -> int:
    try:
        df = pd.read_csv(args.predictions)
    except (OSError, pd.errors.ParserError) as e:
        print(f"ERROR: could not read predictions CSV: {e}", file=sys.stderr)
        return 1
    protected = [a.strip() for a in args.protected_attrs.split(",") if a.strip()]
    rater_cols: list[str] | None = None
    if args.rater_cols:
        rater_cols = [c.strip() for c in args.rater_cols.split(",") if c.strip()]
    errs, resolved_k = _validate_dataframe(df, protected, args.num_classes, rater_cols)
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
        rater_cols=rater_cols,
        rater_missing_value=args.rater_missing_value,
        bootstrap_method=args.bootstrap_method,
        bootstrap_iters=args.bootstrap_iters,
        permutation_iters=args.permutation_iters,
        alpha=args.alpha,
        power=args.power,
    )
    result = write_evidence_pack(df, cfg)
    print(f"OK: wrote evidence pack to {args.output}/")
    print(f"  K (number of classes) = {resolved_k}")
    print(f"  - {result['report_md']}  (sha256={result['report_md_sha256'][:12]}...)")
    print(f"  - {result['evidence_json']}  (sha256={result['evidence_json_sha256'][:12]}...)")
    print(f"  - {result['audit_sha256']}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    labels = [c.strip() for c in args.labels.split(",") if c.strip()]
    if len(labels) != len(args.predictions):
        print(
            f"ERROR: got {len(args.predictions)} predictions paths but {len(labels)} labels.",
            file=sys.stderr,
        )
        return 1
    protected = [a.strip() for a in args.protected_attrs.split(",") if a.strip()]

    dfs: list[pd.DataFrame] = []
    for path in args.predictions:
        try:
            sub_df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as e:
            print(f"ERROR: could not read {path}: {e}", file=sys.stderr)
            return 1
        errs, _ = _validate_dataframe(sub_df, protected, args.num_classes)
        if errs:
            for err in errs:
                print(f"ERROR ({path}): {err}", file=sys.stderr)
            return 1
        dfs.append(sub_df)

    demog_attrs = [a for a in protected if a != args.site_attribute]
    result = compare_models(
        dfs,
        labels,
        site_attribute=args.site_attribute,
        demographic_attributes=demog_attrs,
        num_classes=args.num_classes,
        fairness_floor=args.fairness_floor,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(
        {
            "tool": "fmm-fairness-eval",
            "tool_version": __version__,
            "command": "compare",
            "predictions_files": list(args.predictions),
            "labels": labels,
            "fairness_floor": args.fairness_floor,
            "result": result.to_dict(),
        },
        indent=2,
        sort_keys=True,
    )
    md_text = render_comparison_markdown(result)
    json_path = out / "comparison-evidence.json"
    md_path = out / "comparison-report.md"
    audit_path = out / "audit.sha256"
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    json_sha = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    md_sha = hashlib.sha256(md_text.encode("utf-8")).hexdigest()
    audit_path.write_text(
        f"{json_sha}  comparison-evidence.json\n{md_sha}  comparison-report.md\n",
        encoding="utf-8",
    )
    print(f"OK: wrote comparison pack to {args.output}/")
    print(f"  - {md_path}  (sha256={md_sha[:12]}...)")
    print(f"  - {json_path}  (sha256={json_sha[:12]}...)")
    print(f"  - {audit_path}")
    print(f"  Frontier: {', '.join(result.pareto_frontier_labels) or '(empty)'}")
    if result.recommended_label is not None:
        print(f"  Recommended: {result.recommended_label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "compare":
        return _run_compare(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
