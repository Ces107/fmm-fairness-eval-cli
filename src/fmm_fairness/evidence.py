"""Evidence-pack emitter.

Produces:
- fairness-report.md     human-readable, regulator-friendly
- fairness-evidence.json machine-readable, AI Act Art. 10 compatible
- audit.sha256           SHA-256 of both above, for tamper-evidence

The emitter is K-aware: binary inputs receive the EO / DP / CAL block (the
v0.1 set), and multi-class inputs receive the F1-family block (weighted F1
gap, macro F1 gap, per-class F1 gap, multi-class OVR macro AUC gap). The
inter-site AUC variance is reported for both.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fmm_fairness import __version__
from fmm_fairness.metrics import (
    calibration_gap,
    demographic_parity_gap,
    detect_num_classes,
    equal_opportunity_gap,
    inter_site_auc_variance,
    macro_f1_gap,
    multi_class_auc_gap,
    per_class_f1_gap,
    samd_fairness_score,
    weighted_f1_gap,
)

REPORT_FILENAME = "fairness-report.md"
EVIDENCE_FILENAME = "fairness-evidence.json"
AUDIT_FILENAME = "audit.sha256"


@dataclass
class EvaluationConfig:
    """Inputs to the evidence emitter."""

    predictions_path: str
    protected_attrs: list[str]
    site_attribute: str = "site"
    manifest_mode: str | None = None   # None | "ai-act"
    output_dir: str = "fairness-report"
    timestamp_iso: str | None = None   # injectable for deterministic tests
    num_classes: int | None = None     # explicit K; None => auto-detect


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _isoformat_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _per_attribute_block(
    df: pd.DataFrame, attr: str, K: int
) -> dict[str, Any]:
    """All per-attribute metrics. K-aware (binary vs multi-class)."""
    block: dict[str, Any] = {}
    if K == 2:
        block["equal_opportunity_gap"] = equal_opportunity_gap(df, attr).to_dict()
        block["demographic_parity_gap"] = demographic_parity_gap(df, attr).to_dict()
        block["calibration_gap"] = calibration_gap(df, attr).to_dict()
    block["weighted_f1_gap"] = weighted_f1_gap(df, attr, num_classes=K).to_dict()
    block["macro_f1_gap"] = macro_f1_gap(df, attr, num_classes=K).to_dict()
    block["per_class_f1_gap"] = per_class_f1_gap(df, attr, num_classes=K).to_dict()
    block["multi_class_auc_gap"] = multi_class_auc_gap(df, attr, num_classes=K).to_dict()
    return block


def build_evidence(df: pd.DataFrame, cfg: EvaluationConfig) -> dict[str, Any]:
    """Compute all metrics and assemble the JSON evidence dict."""
    ts = cfg.timestamp_iso or _isoformat_now()
    K = detect_num_classes(df, cfg.num_classes)
    demog_attrs = [a for a in cfg.protected_attrs if a != cfg.site_attribute]
    per_attribute = {a: _per_attribute_block(df, a, K) for a in cfg.protected_attrs}
    site_metric = (
        inter_site_auc_variance(df, cfg.site_attribute, num_classes=K).to_dict()
        if cfg.site_attribute in df.columns
        else None
    )
    composite = samd_fairness_score(
        df,
        site_attribute=cfg.site_attribute,
        demographic_attributes=demog_attrs,
        num_classes=K,
    )

    evidence: dict[str, Any] = {
        "tool": "fmm-fairness-eval",
        "tool_version": __version__,
        "generated_at_utc": ts,
        "predictions_file": cfg.predictions_path,
        "n_samples": len(df),
        "num_classes": K,
        "protected_attributes_declared": cfg.protected_attrs,
        "site_attribute": cfg.site_attribute,
        "manifest_mode": cfg.manifest_mode,
        "per_attribute_metrics": per_attribute,
        "inter_site": site_metric,
        "samd_fairness_score": composite,
    }
    if cfg.manifest_mode == "ai-act":
        evidence["regulatory_mapping"] = _ai_act_mapping(K)
    return evidence


def _ai_act_mapping(num_classes: int) -> dict[str, Any]:
    """Cross-cite metrics to EU AI Act articles."""
    art10_metrics = (
        [
            "equal_opportunity_gap",
            "demographic_parity_gap",
            "calibration_gap",
            "weighted_f1_gap",
            "per_class_f1_gap",
        ]
        if num_classes == 2
        else ["weighted_f1_gap", "macro_f1_gap", "per_class_f1_gap"]
    )
    return {
        "framework": "EU AI Act (Regulation 2024/1689)",
        "articles": [
            {
                "article": "Art. 9",
                "title": "Risk management system",
                "mapped_metrics": [
                    "samd_fairness_score",
                    "inter_site_auc_variance",
                    "weighted_f1_gap",
                ],
                "note": (
                    "Weighted-F1 inter-site gap is the headline residual-risk indicator; "
                    "AUC variance gives the K-agnostic generalisation summary."
                ),
            },
            {
                "article": "Art. 10",
                "title": "Data and data governance",
                "mapped_metrics": art10_metrics,
                "note": (
                    "Per-attribute breakdown evidences Art. 10(2)(f-g) examination of "
                    "biases and shortcomings. Multi-class deployments use the F1 family; "
                    "binary deployments retain EO / DP / CAL."
                ),
            },
            {
                "article": "Art. 15",
                "title": "Accuracy, robustness, cybersecurity",
                "mapped_metrics": ["inter_site_auc_variance", "multi_class_auc_gap"],
                "note": (
                    "Site-level AUC variance + OVR-macro AUC gap evidence performance "
                    "generalisation claims under K-class deployments."
                ),
            },
        ],
    }


def _format_per_group_lines(metric: dict[str, Any], indent: str = "  ") -> list[str]:
    """Shared per-group rendering used across metrics."""
    lines: list[str] = []
    for grp in metric["per_group"]:
        line = f"{indent}- `{grp['group']}` (n={grp['n']}): {grp['value']:.4f}"
        if grp.get("ci_low") is not None:
            line += f"  [bootstrap CI: {grp['ci_low']:.4f}, {grp['ci_high']:.4f}]"
        lines.append(line)
        if grp.get("per_class") is not None:
            pretty = ", ".join(f"c{i}={v:.4f}" for i, v in enumerate(grp["per_class"]))
            lines.append(f"{indent}  per-class F1: {pretty}")
    return lines


def render_markdown(evidence: dict[str, Any]) -> str:
    """Human-readable report. Stable formatting (no model-style flourishes)."""
    lines: list[str] = []
    lines.append("# Fairness evaluation report")
    lines.append("")
    lines.append(f"- **Tool**: {evidence['tool']} v{evidence['tool_version']}")
    lines.append(f"- **Generated (UTC)**: {evidence['generated_at_utc']}")
    lines.append(f"- **Predictions file**: `{evidence['predictions_file']}`")
    lines.append(f"- **Sample count**: {evidence['n_samples']}")
    lines.append(f"- **Number of classes (K)**: {evidence['num_classes']}")
    lines.append(
        f"- **Protected attributes**: {', '.join(evidence['protected_attributes_declared'])}"
    )
    lines.append(f"- **Site attribute**: `{evidence['site_attribute']}`")
    if evidence.get("manifest_mode"):
        lines.append(f"- **Manifest mode**: `{evidence['manifest_mode']}`")
    lines.append("")
    lines.append("## SaMD composite fairness score")
    comp = evidence["samd_fairness_score"]
    lines.append(
        f"- **Score**: {comp['samd_fairness_score']:.4f}  "
        f"(1.0 = perfectly fair, 0.0 = maximally unfair)"
    )
    components = comp["components"]
    lines.append(
        "- **Components**: "
        f"F1_site={components['f1_site_term']:.4f}, "
        f"site_term={components['site_term']:.4f}, "
        f"EO={components['eo_mean']:.4f}, "
        f"DP={components['dp_mean']:.4f}, "
        f"CAL={components['cal_mean']:.4f}"
    )
    lines.append(f"- **Effective weights**: {comp['weights']}")
    lines.append("")
    if evidence.get("inter_site"):
        lines.append("## Inter-site AUC")
        site = evidence["inter_site"]
        lines.append(f"- **Variance across sites**: {site['gap']:.6f}")
        for grp in site["per_group"]:
            lines.append(f"  - `{grp['group']}` (n={grp['n']}): AUC={grp['value']:.4f}")
        if site["excluded_groups"]:
            lines.append(f"- **Excluded (n<min)**: {site['excluded_groups']}")
        lines.append("")
    lines.append("## Per-attribute fairness gaps")
    for attr, metrics in evidence["per_attribute_metrics"].items():
        lines.append(f"### `{attr}`")
        for mname, m in metrics.items():
            ci = ""
            if m.get("gap_ci_low") is not None:
                ci = f"  [95% CI: {m['gap_ci_low']:.4f}, {m['gap_ci_high']:.4f}]"
            extra = ""
            if m.get("per_class_gap") is not None:
                pretty = ", ".join(f"c{i}={v:.4f}" for i, v in enumerate(m["per_class_gap"]))
                extra = f"  (per-class gaps: {pretty})"
            lines.append(f"- **{mname}**: {m['gap']:.4f}{ci}{extra}")
            lines.extend(_format_per_group_lines(m))
            if m["excluded_groups"]:
                lines.append(f"  - Excluded (n<min): {m['excluded_groups']}")
        lines.append("")
    if evidence.get("regulatory_mapping"):
        lines.append("## Regulatory mapping")
        mapping = evidence["regulatory_mapping"]
        lines.append(f"- **Framework**: {mapping['framework']}")
        for art in mapping["articles"]:
            lines.append(f"  - **{art['article']} — {art['title']}**")
            lines.append(f"    - Metrics: {', '.join(art['mapped_metrics'])}")
            lines.append(f"    - Note: {art['note']}")
        lines.append("")
    lines.append("## Caveats (read before quoting these numbers)")
    lines.append("- Gap metrics depend on the operating threshold (or argmax decision) used to produce y_pred.")
    lines.append("- Groups with n < min_group_n were excluded; small-sample bootstrap CIs are approximate.")
    lines.append("- Multi-site bias is often confounded with prevalence shift across sites.")
    lines.append("- Per-class F1 gaps identify which class drives an inter-site disparity; combine with the per-class breakdown in the per-group rows.")
    lines.append("- This tool measures fairness; it does not certify it.")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_evidence_pack(
    df: pd.DataFrame, cfg: EvaluationConfig
) -> dict[str, str]:
    """Emit JSON + Markdown + audit SHA-256. Return paths + signatures."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    evidence = build_evidence(df, cfg)
    json_text = json.dumps(evidence, indent=2, sort_keys=True)
    md_text = render_markdown(evidence)

    json_path = out / EVIDENCE_FILENAME
    md_path = out / REPORT_FILENAME
    audit_path = out / AUDIT_FILENAME

    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")

    json_sha = _sha256_text(json_text)
    md_sha = _sha256_text(md_text)
    audit_text = (
        f"{json_sha}  {EVIDENCE_FILENAME}\n"
        f"{md_sha}  {REPORT_FILENAME}\n"
    )
    audit_path.write_text(audit_text, encoding="utf-8")

    return {
        "evidence_json": str(json_path),
        "report_md": str(md_path),
        "audit_sha256": str(audit_path),
        "evidence_json_sha256": json_sha,
        "report_md_sha256": md_sha,
    }
