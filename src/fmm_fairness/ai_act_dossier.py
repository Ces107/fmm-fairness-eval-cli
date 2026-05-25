"""EU AI Act dossier builder — Art. 9 / 10 / 13 / 14 / 15 / 72 cross-mapping.

The basic `--manifest-mode ai-act` block (S2..S6 era) maps the numeric
metrics to four Art. 9 / 10 / 14 / 15 entries. This module extends the
mapping to the full provider-facing dossier shape:

- Art. 9  — Risk management system (already covered; carried forward)
- Art. 10 — Data and data governance (already covered; carried forward)
- Art. 13 — Transparency / information to deployers (NEW, model-card driven)
- Art. 14 — Human oversight (consolidated with kappa metrics from S2)
- Art. 15 — Accuracy, robustness, cybersecurity (extended in S6)
- Art. 72 — Post-market monitoring (NEW, template-driven)

The numeric content stays in the same metric blocks; this module attaches
the cross-references, the model-card payload (when provided via
``--model-card``), and the post-market-monitoring schema/template path.

Conventions
-----------
- Unknown YAML / dict fields in the model card pass through untouched.
- Missing model card is OK: ``ai_act_full.model_card`` becomes ``None``
  and the Art. 13 block notes that the dossier is incomplete.
- This module never embeds the entire template file content; it embeds the
  bundled template version and the relative path so the dossier stays
  small. Auditors read the templates alongside the JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEMPLATES_RELATIVE_PATH = "fmm_fairness/templates/ai_act"
TEMPLATE_FILES = (
    "model-card.yaml",
    "human-oversight.md",
    "post-market-monitoring.csv",
    "post-market-monitoring.md",
    "README.md",
)
TEMPLATE_PACK_VERSION = "1.0.0"

# Drift criteria used by Art. 72 monitoring; these are bundled defaults
# overridable through the model card under ``drift_thresholds``.
DEFAULT_DRIFT_THRESHOLDS: dict[str, float] = {
    "weighted_f1_gap_absolute": 0.10,
    "brier_absolute": 0.02,
    "ai_vs_pooled_kappa_drop": 0.10,
    "hosmer_lemeshow_p_floor": 0.01,
}


@dataclass
class AiActFullConfig:
    """Configuration overrides for the ai-act-full manifest."""

    model_card_path: str | None = None
    template_pack_path: str | None = None  # override to a forked template dir
    drift_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DRIFT_THRESHOLDS)
    )


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON file as a dict at the top level."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: expected a JSON object at the top level.")
        return loaded
    import yaml

    loaded_yaml = yaml.safe_load(text)
    if loaded_yaml is None:
        return {}
    if not isinstance(loaded_yaml, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level.")
    return loaded_yaml


def load_model_card(path: str | Path) -> dict[str, Any]:
    """Load a model card from YAML / JSON. Returns ``{}`` for missing path."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model card not found: {p}")
    return _load_yaml_or_json(p)


def _template_pack_info(cfg: AiActFullConfig) -> dict[str, Any]:
    base = Path(cfg.template_pack_path) if cfg.template_pack_path else None
    return {
        "version": TEMPLATE_PACK_VERSION,
        "relative_path": TEMPLATES_RELATIVE_PATH if base is None else str(base),
        "files": list(TEMPLATE_FILES),
    }


def _art_13_block(model_card: dict[str, Any], cfg: AiActFullConfig) -> dict[str, Any]:
    return {
        "article": "Art. 13",
        "title": "Transparency and provision of information to deployers",
        "model_card_present": bool(model_card),
        "model_card_path": cfg.model_card_path,
        "model_card": model_card or None,
        "template": "fmm_fairness/templates/ai_act/model-card.yaml",
        "note": (
            "Art. 13 requires the provider to deliver instructions for use "
            "and a transparency notice; the model-card YAML is the dossier "
            "shape and the evidence pack (gap + CI + p-value + kappa + Brier) "
            "is the numeric content."
        ),
    }


def _art_14_block(num_classes: int, has_raters: bool) -> dict[str, Any]:
    citations = [
        "ai_vs_pooled_raters_kappa",
        "cohen_kappa_matrix",
        "fleiss_kappa",
        "krippendorff_alpha",
    ]
    note = (
        "The pooled-rater Cohen kappa (and its bootstrap CI) is the headline "
        "Art. 14 metric: it places the AI against the human-expert reference. "
        "The kappa matrix documents the level of disagreement among the human "
        "raters themselves; that is the floor of what the AI can be expected "
        "to inherit. See fmm_fairness/templates/ai_act/human-oversight.md for the matching "
        "procedure template."
    )
    if not has_raters:
        note = (
            "No --rater-cols was provided; the Art. 14 block is documentary "
            "only. To make Art. 14 numerically defensible, re-run with "
            "--rater-cols pointing at the human-reviewer columns."
        )
    return {
        "article": "Art. 14",
        "title": "Human oversight",
        "mapped_metrics": citations,
        "has_rater_evidence": has_raters,
        "template": "fmm_fairness/templates/ai_act/human-oversight.md",
        "note": note,
    }


def _art_72_block(cfg: AiActFullConfig) -> dict[str, Any]:
    return {
        "article": "Art. 72",
        "title": "Post-market monitoring",
        "csv_schema_template": "fmm_fairness/templates/ai_act/post-market-monitoring.csv",
        "procedure_template": "fmm_fairness/templates/ai_act/post-market-monitoring.md",
        "drift_thresholds": dict(cfg.drift_thresholds),
        "note": (
            "Re-run the CLI on the appended post-market-monitoring CSV; the "
            "fresh evidence pack is diffed against the reference pack using "
            "the documented drift thresholds. A breach triggers an Art. 72 "
            "incident report appended to the technical-documentation file "
            "(Annex IV §7)."
        ),
    }


def _art_9_block_full() -> dict[str, Any]:
    return {
        "article": "Art. 9",
        "title": "Risk management system",
        "mapped_metrics": [
            "samd_fairness_score",
            "inter_site_auc_variance",
            "weighted_f1_gap",
            "intersectional_weighted_f1_gap",
        ],
        "note": (
            "Risk management is iterative; the headline residual-risk "
            "indicator is the inter-site weighted-F1 gap with its BCa CI. "
            "Intersectional gaps surface compounding disparities that the "
            "single-axis indicator can hide."
        ),
    }


def _art_10_block_full(num_classes: int) -> dict[str, Any]:
    metrics_multi = [
        "weighted_f1_gap",
        "macro_f1_gap",
        "per_class_f1_gap",
        "intersectional_weighted_f1_gap",
        "intersectional_macro_f1_gap",
    ]
    metrics_binary = [
        "equal_opportunity_gap",
        "demographic_parity_gap",
        "calibration_gap",
        "weighted_f1_gap",
        "per_class_f1_gap",
        "intersectional_weighted_f1_gap",
    ]
    mapped = metrics_binary if num_classes == 2 else metrics_multi
    return {
        "article": "Art. 10",
        "title": "Data and data governance",
        "mapped_metrics": mapped,
        "note": (
            "Per-attribute breakdown evidences Art. 10(2)(f-g) examination "
            "of biases and shortcomings. Multi-class deployments use the F1 "
            "family; binary deployments retain EO / DP / CAL. "
            "Intersectional cross-products evidence Art. 10(2)(g) detection "
            "of biases that affect persons due to a combination of "
            "characteristics."
        ),
    }


def _art_15_block_full() -> dict[str, Any]:
    return {
        "article": "Art. 15",
        "title": "Accuracy, robustness, cybersecurity",
        "mapped_metrics": [
            "inter_site_auc_variance",
            "multi_class_auc_gap",
            "brier_score",
            "hosmer_lemeshow",
            "permutation_p_value",
            "minimum_detectable_effect",
        ],
        "note": (
            "Discrimination generalisation (AUC variance + OVR-macro AUC "
            "gap) joins probability-calibration robustness (multi-class "
            "Brier + per-class Hosmer-Lemeshow). The permutation p-value "
            "and the minimum-detectable-effect bound the inferential claim "
            "about any observed inter-site gap."
        ),
    }


def build_ai_act_full_block(
    num_classes: int,
    has_raters: bool,
    cfg: AiActFullConfig,
    model_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ai_act_full block emitted alongside `regulatory_mapping`.

    ``model_card`` is the parsed model-card dict (or ``None``); pass ``{}``
    for an empty card.
    """
    mc = model_card or {}
    return {
        "framework": "EU AI Act (Regulation 2024/1689)",
        "manifest_mode": "ai-act-full",
        "template_pack": _template_pack_info(cfg),
        "articles": [
            _art_9_block_full(),
            _art_10_block_full(num_classes),
            _art_13_block(mc, cfg),
            _art_14_block(num_classes, has_raters),
            _art_15_block_full(),
            _art_72_block(cfg),
        ],
    }


def find_bundled_template_path() -> Path:
    """Locate the bundled templates directory next to the installed package.

    The templates live inside the package at
    ``fmm_fairness/templates/ai_act/``. This function returns the absolute
    path to that directory, working identically in source-checkout and in
    installed-wheel layouts.
    """
    in_package = Path(__file__).resolve().parent / "templates" / "ai_act"
    if in_package.is_dir():
        return in_package
    # Last-resort fallback: relative path for tooling that resolves later.
    return Path(TEMPLATES_RELATIVE_PATH)


def list_bundled_template_files() -> list[str]:
    """Return the names of bundled templates the dossier references."""
    path = find_bundled_template_path()
    present: list[str] = []
    for name in TEMPLATE_FILES:
        if (path / name).is_file():
            present.append(name)
    return present
