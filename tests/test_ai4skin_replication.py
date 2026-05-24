"""CI gate for the AI4SkIN golden replication example.

Reproduces the smoke run that the notebook in
``examples/ai4skin-replication/replicate.ipynb`` performs, but as a
pytest case so the gate fires on every commit without needing
``nbconvert`` in the test image.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "ai4skin-replication"

TARGET_WEIGHTED_GAP = 0.1657
TOL_WEIGHTED_GAP = 0.005


@pytest.fixture(scope="module")
def evidence_pack(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    out_dir = tmp_path_factory.mktemp("ai4skin-out")
    # Regenerate the CSV in-place — byte-deterministic given the seed.
    subprocess.run(
        [sys.executable, str(EXAMPLE_DIR / "build_dataset.py")],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "fmm_fairness.cli", "evaluate",
            str(EXAMPLE_DIR / "predictions.csv"),
            "--protected-attrs", "site",
            "--site-attribute", "site",
            "--rater-cols", "doc1,doc2,doc3,doc4,doc5,doc6,doc7,doc8,doc9,doc10",
            "--bootstrap-method", "bca",
            "--bootstrap-iters", "2000",
            "--permutation-iters", "2000",
            "--output", str(out_dir),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads((out_dir / "fairness-evidence.json").read_text(encoding="utf-8"))


def test_predictions_csv_matches_published_n(evidence_pack: dict[str, object]) -> None:
    assert evidence_pack["n_samples"] == 157, (
        "Replication CSV must contain exactly 157 rows (41 HCUV + 116 HUSC)."
    )


def test_weighted_f1_gap_reproduces_table_6(evidence_pack: dict[str, object]) -> None:
    wf1 = evidence_pack["per_attribute_metrics"]["site"]["weighted_f1_gap"]
    assert abs(wf1["gap"] - TARGET_WEIGHTED_GAP) <= TOL_WEIGHTED_GAP, (
        f"weighted_f1_gap drifted: got {wf1['gap']:.4f}, "
        f"target {TARGET_WEIGHTED_GAP} ± {TOL_WEIGHTED_GAP}"
    )
    per_group = {g["group"]: g["value"] for g in wf1["per_group"]}
    assert abs(per_group["HUSC"] - 0.9224) <= 0.005
    assert abs(per_group["HCUV"] - 0.7567) <= 0.005


def test_permutation_p_value_significant(evidence_pack: dict[str, object]) -> None:
    wf1 = evidence_pack["per_attribute_metrics"]["site"]["weighted_f1_gap"]
    p = wf1["permutation_p_value"]
    assert p is not None and p < 0.05, (
        f"Permutation p-value should reject H0 (no gap); got {p}"
    )


def test_mde_below_observed_gap(evidence_pack: dict[str, object]) -> None:
    wf1 = evidence_pack["per_attribute_metrics"]["site"]["weighted_f1_gap"]
    mde = wf1["minimum_detectable_effect"]
    assert mde is not None and mde > 0, "MDE should be a positive scalar"
    assert mde < 0.30, f"MDE@80%power unexpectedly large: {mde:.4f}"


def test_inter_rater_kappa_in_plausible_band(evidence_pack: dict[str, object]) -> None:
    kappa = evidence_pack["inter_rater_agreement"]["ai_vs_pooled_raters_kappa"]["value"]
    assert 0.70 < kappa < 0.90, (
        f"AI-vs-pooled κ {kappa:.4f} outside the published-plausible 0.70-0.90 band; "
        "the rater synthesis in build_dataset.py may have drifted."
    )
