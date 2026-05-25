"""Tests for fmm_fairness.ai_act_dossier."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fmm_fairness.ai_act_dossier import (
    DEFAULT_DRIFT_THRESHOLDS,
    TEMPLATE_FILES,
    TEMPLATE_PACK_VERSION,
    TEMPLATES_RELATIVE_PATH,
    AiActFullConfig,
    build_ai_act_full_block,
    find_bundled_template_path,
    list_bundled_template_files,
    load_model_card,
)

TEMPLATE_ROOT = find_bundled_template_path()


class TestTemplatesPresent:
    def test_template_dir_found_relative_to_package(self) -> None:
        assert TEMPLATE_ROOT.is_dir(), f"templates dir missing: {TEMPLATE_ROOT}"

    def test_all_expected_template_files_present(self) -> None:
        present = list_bundled_template_files()
        for name in TEMPLATE_FILES:
            assert name in present, f"missing template file: {name}"

    def test_post_market_monitoring_csv_is_valid(self) -> None:
        pmm = TEMPLATE_ROOT / "post-market-monitoring.csv"
        with pmm.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        required = {
            "sample_id",
            "site",
            "timestamp_utc",
            "y_true",
            "y_pred",
            "y_score_0",
            "y_score_1",
            "y_score_2",
            "y_score_3",
            "y_score_4",
            "y_score_5",
            "reviewer_id",
            "reviewer_label",
            "confidence_flag",
            "subgroup_flag",
            "decision_outcome",
            "decision_modified_by_human",
            "notes",
        }
        assert reader.fieldnames is not None
        actual = set(reader.fieldnames)
        assert required.issubset(actual), (
            f"PMM CSV is missing required columns: {required - actual}"
        )
        assert len(rows) >= 1


class TestModelCardLoader:
    def test_loads_bundled_model_card(self) -> None:
        mc = load_model_card(TEMPLATE_ROOT / "model-card.yaml")
        for key in (
            "system_identification",
            "intended_purpose",
            "high_risk_class",
            "performance_summary",
            "oversight_summary",
        ):
            assert key in mc

    def test_missing_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_card("/does/not/exist.yaml")

    def test_empty_path_returns_empty_dict(self) -> None:
        assert load_model_card("") == {}

    def test_loads_json(self, tmp_path: Path) -> None:
        p = tmp_path / "mc.json"
        p.write_text(json.dumps({"system_identification": {"name": "foo"}}))
        out = load_model_card(p)
        assert out["system_identification"]["name"] == "foo"

    def test_loads_yaml_with_lists(self) -> None:
        # The bundled template includes list-of-mapping under residual_risks.
        mc = load_model_card(TEMPLATE_ROOT / "model-card.yaml")
        assert isinstance(mc["residual_risks"], list)
        assert mc["residual_risks"][0]["id"] == "RR-1"
        assert isinstance(mc["limitations"], list)


class TestBuildAiActFullBlock:
    def test_articles_present(self) -> None:
        cfg = AiActFullConfig()
        block = build_ai_act_full_block(num_classes=6, has_raters=True, cfg=cfg)
        articles = {a["article"] for a in block["articles"]}
        assert articles == {"Art. 9", "Art. 10", "Art. 13", "Art. 14", "Art. 15", "Art. 72"}
        assert block["manifest_mode"] == "ai-act-full"

    def test_template_pack_metadata(self) -> None:
        cfg = AiActFullConfig()
        block = build_ai_act_full_block(num_classes=6, has_raters=True, cfg=cfg)
        tp = block["template_pack"]
        assert tp["version"] == TEMPLATE_PACK_VERSION
        assert tp["relative_path"] == TEMPLATES_RELATIVE_PATH
        for name in TEMPLATE_FILES:
            assert name in tp["files"]

    def test_art_13_carries_model_card_payload(self) -> None:
        cfg = AiActFullConfig(model_card_path="path/to/card.yaml")
        mc = {"system_identification": {"name": "X"}}
        block = build_ai_act_full_block(
            num_classes=6, has_raters=True, cfg=cfg, model_card=mc
        )
        art13 = next(a for a in block["articles"] if a["article"] == "Art. 13")
        assert art13["model_card_present"] is True
        assert art13["model_card"] == mc
        assert art13["model_card_path"] == "path/to/card.yaml"

    def test_art_13_without_model_card_marks_incomplete(self) -> None:
        cfg = AiActFullConfig()
        block = build_ai_act_full_block(num_classes=6, has_raters=True, cfg=cfg)
        art13 = next(a for a in block["articles"] if a["article"] == "Art. 13")
        assert art13["model_card_present"] is False
        assert art13["model_card"] is None

    def test_art_14_flag_when_no_raters(self) -> None:
        cfg = AiActFullConfig()
        block = build_ai_act_full_block(num_classes=6, has_raters=False, cfg=cfg)
        art14 = next(a for a in block["articles"] if a["article"] == "Art. 14")
        assert art14["has_rater_evidence"] is False
        # Note explains the absence
        assert "rater-cols" in art14["note"]

    def test_art_72_carries_drift_thresholds(self) -> None:
        cfg = AiActFullConfig()
        block = build_ai_act_full_block(num_classes=6, has_raters=True, cfg=cfg)
        art72 = next(a for a in block["articles"] if a["article"] == "Art. 72")
        assert art72["drift_thresholds"] == DEFAULT_DRIFT_THRESHOLDS
        assert art72["csv_schema_template"].endswith("post-market-monitoring.csv")

    def test_art_72_drift_thresholds_override(self) -> None:
        cfg = AiActFullConfig(
            drift_thresholds={
                "weighted_f1_gap_absolute": 0.05,
                "brier_absolute": 0.01,
                "ai_vs_pooled_kappa_drop": 0.05,
                "hosmer_lemeshow_p_floor": 0.05,
            }
        )
        block = build_ai_act_full_block(num_classes=6, has_raters=True, cfg=cfg)
        art72 = next(a for a in block["articles"] if a["article"] == "Art. 72")
        assert art72["drift_thresholds"]["weighted_f1_gap_absolute"] == 0.05

    def test_binary_vs_multiclass_metrics(self) -> None:
        cfg = AiActFullConfig()
        binary = build_ai_act_full_block(num_classes=2, has_raters=False, cfg=cfg)
        multi = build_ai_act_full_block(num_classes=6, has_raters=False, cfg=cfg)
        binary_art10 = next(a for a in binary["articles"] if a["article"] == "Art. 10")
        multi_art10 = next(a for a in multi["articles"] if a["article"] == "Art. 10")
        assert "equal_opportunity_gap" in binary_art10["mapped_metrics"]
        assert "equal_opportunity_gap" not in multi_art10["mapped_metrics"]
        assert "weighted_f1_gap" in multi_art10["mapped_metrics"]
