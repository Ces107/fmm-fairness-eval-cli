"""End-to-end CLI test for multi-class inputs (K = 6 AI4SkIN-shaped fixture)."""
from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fmm_fairness.cli import main
from fmm_fairness.evidence import EvaluationConfig, build_evidence, write_evidence_pack
from tests.test_multi_class import _make_ai4skin_shaped_df


class TestCliMultiClassEndToEnd(unittest.TestCase):
    def test_evaluate_writes_evidence_pack_with_multi_class_block(self) -> None:
        df = _make_ai4skin_shaped_df()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "preds.csv"
            df.to_csv(csv_path, index=False)
            out_dir = Path(tmp) / "report"
            rc = main(
                [
                    "evaluate",
                    str(csv_path),
                    "--protected-attrs",
                    "site,sex",
                    "--site-attribute",
                    "site",
                    "--num-classes",
                    "6",
                    "--output",
                    str(out_dir),
                ]
            )
            self.assertEqual(rc, 0)
            evidence_json = json.loads(
                (out_dir / "fairness-evidence.json").read_text(encoding="utf-8")
            )
            self.assertEqual(evidence_json["num_classes"], 6)
            per_site = evidence_json["per_attribute_metrics"]["site"]
            self.assertIn("weighted_f1_gap", per_site)
            self.assertIn("per_class_f1_gap", per_site)
            self.assertNotIn("equal_opportunity_gap", per_site)  # binary-only
            self.assertIsNotNone(per_site["per_class_f1_gap"].get("per_class_gap"))
            self.assertEqual(
                len(per_site["per_class_f1_gap"]["per_class_gap"]), 6
            )

    def test_evaluate_rejects_score_outside_unit_interval(self) -> None:
        df = _make_ai4skin_shaped_df(n_per_site=50)
        df.loc[0, "y_score_0"] = 1.5
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "preds.csv"
            df.to_csv(csv_path, index=False)
            with patch("sys.stderr", new_callable=StringIO) as err:
                rc = main(
                    [
                        "evaluate",
                        str(csv_path),
                        "--protected-attrs",
                        "site",
                        "--output",
                        str(Path(tmp) / "out"),
                    ]
                )
            self.assertEqual(rc, 1)
            self.assertIn("Multi-class score columns", err.getvalue())

    def test_evaluate_rejects_label_outside_k(self) -> None:
        df = _make_ai4skin_shaped_df(n_per_site=50)
        df.loc[0, "y_true"] = 9  # outside [0, K-1] for K=6
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "preds.csv"
            df.to_csv(csv_path, index=False)
            with patch("sys.stderr", new_callable=StringIO) as err:
                rc = main(
                    [
                        "evaluate",
                        str(csv_path),
                        "--protected-attrs",
                        "site",
                        "--num-classes",
                        "6",
                        "--output",
                        str(Path(tmp) / "out"),
                    ]
                )
            self.assertEqual(rc, 1)
            self.assertIn("class indices outside", err.getvalue())


class TestEvidenceMultiClassRendering(unittest.TestCase):
    def test_markdown_includes_k_and_per_class_pretty_print(self) -> None:
        df = _make_ai4skin_shaped_df()
        cfg = EvaluationConfig(
            predictions_path="x.csv",
            protected_attrs=["site"],
            site_attribute="site",
            timestamp_iso="2026-05-21T00:00:00Z",
        )
        e = build_evidence(df, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            cfg2 = EvaluationConfig(
                predictions_path="x.csv",
                protected_attrs=["site"],
                site_attribute="site",
                output_dir=tmp,
                timestamp_iso="2026-05-21T00:00:00Z",
            )
            res = write_evidence_pack(df, cfg2)
            md = Path(res["report_md"]).read_text(encoding="utf-8")
        self.assertIn("Number of classes (K)", md)
        self.assertIn("weighted_f1_gap", md)
        self.assertIn("per-class F1", md)
        # AI Act dossier picks K-specific Art. 10 metric list.
        self.assertEqual(e["num_classes"], 6)


class TestBinaryCsvBackwardCompat(unittest.TestCase):
    """A v0.1 binary CSV (only y_score, no y_score_0/1) must still flow."""

    def test_binary_csv_still_works_without_num_classes_flag(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        y_true = rng.integers(0, 2, size=n)
        y_score = np.where(
            y_true == 1, rng.beta(7, 2, size=n), rng.beta(2, 7, size=n)
        )
        y_pred = (y_score >= 0.5).astype(int)
        df = pd.DataFrame(
            {
                "y_true": y_true.astype(int),
                "y_pred": y_pred.astype(int),
                "y_score": y_score,
                "site": rng.choice(["A", "B"], size=n),
                "sex": rng.choice(["F", "M"], size=n),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "preds.csv"
            df.to_csv(csv_path, index=False)
            rc = main(
                [
                    "evaluate",
                    str(csv_path),
                    "--protected-attrs",
                    "site,sex",
                    "--output",
                    str(Path(tmp) / "out"),
                ]
            )
            self.assertEqual(rc, 0)
            evidence_json = json.loads(
                (Path(tmp) / "out" / "fairness-evidence.json").read_text(encoding="utf-8")
            )
            self.assertEqual(evidence_json["num_classes"], 2)
            per_site = evidence_json["per_attribute_metrics"]["site"]
            self.assertIn("equal_opportunity_gap", per_site)
            self.assertIn("weighted_f1_gap", per_site)


if __name__ == "__main__":
    unittest.main()
