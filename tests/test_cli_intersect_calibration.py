"""End-to-end CLI tests for the S6 intersect + calibration features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fmm_fairness.cli import main


def _write_synth_csv(path: Path, n: int = 200, seed: int = 20260525) -> None:
    rng = np.random.default_rng(seed)
    site = rng.choice(["A", "B"], size=n)
    sex = rng.choice(["M", "F"], size=n)
    y_true = rng.integers(0, 3, size=n)
    y_pred = y_true.copy()
    flip = (site == "A") & (sex == "F")
    y_pred[flip] = (y_pred[flip] + 1) % 3
    scores = np.full((n, 3), 0.2)
    scores[np.arange(n), y_pred] = 0.6
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score_0": scores[:, 0],
            "y_score_1": scores[:, 1],
            "y_score_2": scores[:, 2],
            "site": site,
            "sex": sex,
        }
    )
    df.to_csv(path, index=False)


def test_cli_intersect_emits_intersectional_breakdown(tmp_path: Path) -> None:
    csv = tmp_path / "predictions.csv"
    out = tmp_path / "report"
    _write_synth_csv(csv)
    rc = main(
        [
            "evaluate",
            str(csv),
            "--protected-attrs",
            "site,sex",
            "--intersect",
            "site*sex",
            "--output",
            str(out),
            "--bootstrap-iters",
            "50",
        ]
    )
    assert rc == 0
    pack = json.loads((out / "fairness-evidence.json").read_text())
    assert "intersectional_breakdown" in pack
    breakdown = pack["intersectional_breakdown"]
    assert breakdown["intersections_declared"] == [["site", "sex"]]
    assert len(breakdown["results"]) == 1
    result = breakdown["results"][0]
    cells = {c["group"] for c in result["weighted_f1_gap"]["per_cell"]}
    assert cells == {"A*F", "A*M", "B*F", "B*M"}
    # The headline disparity is the A*F cell vs the rest.
    assert result["weighted_f1_gap"]["gap"] > 0.3


def test_cli_calibration_block_always_emitted(tmp_path: Path) -> None:
    csv = tmp_path / "predictions.csv"
    out = tmp_path / "report"
    _write_synth_csv(csv)
    rc = main(
        [
            "evaluate",
            str(csv),
            "--protected-attrs",
            "site,sex",
            "--output",
            str(out),
            "--bootstrap-iters",
            "50",
        ]
    )
    assert rc == 0
    pack = json.loads((out / "fairness-evidence.json").read_text())
    assert "calibration" in pack
    cb = pack["calibration"]
    assert "global_brier_score" in cb
    assert set(cb["per_attribute"]) == {"site", "sex"}


def test_cli_render_plots_no_op_when_matplotlib_missing(tmp_path: Path) -> None:
    """--render-plots must not crash if matplotlib is unavailable.

    The renderer warns and returns []; the evidence pack still emits.
    """
    csv = tmp_path / "predictions.csv"
    out = tmp_path / "report"
    _write_synth_csv(csv)
    rc = main(
        [
            "evaluate",
            str(csv),
            "--protected-attrs",
            "site",
            "--output",
            str(out),
            "--render-plots",
            "--bootstrap-iters",
            "50",
        ]
    )
    assert rc == 0
    pack = json.loads((out / "fairness-evidence.json").read_text())
    # rendered_plots is optional; present only when matplotlib succeeded
    assert "rendered_plots" not in pack or isinstance(pack["rendered_plots"], list)
