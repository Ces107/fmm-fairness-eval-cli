"""End-to-end CLI tests for --manifest-mode ai-act-full."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fmm_fairness.cli import main


def _write_predictions_csv(path: Path, n: int = 240, seed: int = 20260525) -> None:
    rng = np.random.default_rng(seed)
    site = rng.choice(["HUSC", "HCUV"], size=n)
    sex = rng.choice(["F", "M"], size=n)
    y_true = rng.integers(0, 6, size=n)
    y_pred = y_true.copy()
    flip = (site == "HCUV") & (sex == "F")
    y_pred[flip] = (y_pred[flip] + 1) % 6
    scores = np.full((n, 6), 0.05)
    scores[np.arange(n), y_pred] = 0.75
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            **{f"y_score_{k}": scores[:, k] for k in range(6)},
            "site": site,
            "sex": sex,
        }
    )
    df.to_csv(path, index=False)


def test_cli_ai_act_full_emits_full_block_with_model_card(tmp_path: Path) -> None:
    csv_path = tmp_path / "predictions.csv"
    out = tmp_path / "report"
    _write_predictions_csv(csv_path)
    # Use the bundled model card from the package
    from fmm_fairness.ai_act_dossier import find_bundled_template_path
    mc_path = find_bundled_template_path() / "model-card.yaml"
    assert mc_path.exists(), "bundled model-card.yaml missing"
    rc = main(
        [
            "evaluate",
            str(csv_path),
            "--protected-attrs",
            "site,sex",
            "--manifest-mode",
            "ai-act-full",
            "--model-card",
            str(mc_path),
            "--output",
            str(out),
            "--bootstrap-iters",
            "50",
        ]
    )
    assert rc == 0
    pack = json.loads((out / "fairness-evidence.json").read_text())
    # Both blocks present: basic regulatory_mapping + ai_act_full
    assert "regulatory_mapping" in pack
    assert "ai_act_full" in pack
    full = pack["ai_act_full"]
    articles = {a["article"] for a in full["articles"]}
    assert articles == {"Art. 9", "Art. 10", "Art. 13", "Art. 14", "Art. 15", "Art. 72"}
    art13 = next(a for a in full["articles"] if a["article"] == "Art. 13")
    assert art13["model_card_present"] is True
    assert art13["model_card"] is not None


def test_cli_ai_act_full_without_model_card_marks_incomplete(tmp_path: Path) -> None:
    csv_path = tmp_path / "predictions.csv"
    out = tmp_path / "report"
    _write_predictions_csv(csv_path)
    rc = main(
        [
            "evaluate",
            str(csv_path),
            "--protected-attrs",
            "site,sex",
            "--manifest-mode",
            "ai-act-full",
            "--output",
            str(out),
            "--bootstrap-iters",
            "50",
        ]
    )
    assert rc == 0
    pack = json.loads((out / "fairness-evidence.json").read_text())
    art13 = next(a for a in pack["ai_act_full"]["articles"] if a["article"] == "Art. 13")
    assert art13["model_card_present"] is False
    assert art13["model_card"] is None


def test_cli_ai_act_full_art_14_no_raters_note(tmp_path: Path) -> None:
    csv_path = tmp_path / "predictions.csv"
    out = tmp_path / "report"
    _write_predictions_csv(csv_path)
    rc = main(
        [
            "evaluate",
            str(csv_path),
            "--protected-attrs",
            "site,sex",
            "--manifest-mode",
            "ai-act-full",
            "--output",
            str(out),
            "--bootstrap-iters",
            "50",
        ]
    )
    assert rc == 0
    pack = json.loads((out / "fairness-evidence.json").read_text())
    art14 = next(a for a in pack["ai_act_full"]["articles"] if a["article"] == "Art. 14")
    assert art14["has_rater_evidence"] is False
    assert "rater-cols" in art14["note"]


def test_cli_ai_act_full_template_pack_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "predictions.csv"
    out = tmp_path / "report"
    _write_predictions_csv(csv_path)
    rc = main(
        [
            "evaluate",
            str(csv_path),
            "--protected-attrs",
            "site",
            "--manifest-mode",
            "ai-act-full",
            "--output",
            str(out),
            "--bootstrap-iters",
            "50",
        ]
    )
    assert rc == 0
    pack = json.loads((out / "fairness-evidence.json").read_text())
    tp = pack["ai_act_full"]["template_pack"]
    assert "version" in tp
    assert tp["relative_path"] == "fmm_fairness/templates/ai_act"
    assert "model-card.yaml" in tp["files"]
    assert "post-market-monitoring.csv" in tp["files"]


def test_cli_manifest_mode_ai_act_basic_still_works(tmp_path: Path) -> None:
    """The pre-S7 ai-act mode is unchanged: regulatory_mapping present,
    no ai_act_full block leaked into the evidence pack."""
    csv_path = tmp_path / "predictions.csv"
    out = tmp_path / "report"
    _write_predictions_csv(csv_path)
    rc = main(
        [
            "evaluate",
            str(csv_path),
            "--protected-attrs",
            "site",
            "--manifest-mode",
            "ai-act",
            "--output",
            str(out),
            "--bootstrap-iters",
            "50",
        ]
    )
    assert rc == 0
    pack = json.loads((out / "fairness-evidence.json").read_text())
    assert "regulatory_mapping" in pack
    assert "ai_act_full" not in pack
