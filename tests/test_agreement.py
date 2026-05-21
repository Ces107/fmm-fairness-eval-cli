"""Tests for fmm_fairness.agreement.

Validation strategy:
1. Identity cases: all raters agree => kappa = alpha = 1.0; complete random
   independence => kappa, alpha close to 0.
2. Textbook references: a 14-subject 6-rater Fleiss-kappa case taken from the
   canonical reference; a small Krippendorff alpha case with a missing rating
   that has a hand-verifiable answer.
3. CLI plumbing: --rater-cols runs end-to-end on the AI4SkIN-shaped fixture
   with synthetic doc1..doc10 columns and a non-trivial kappa.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fmm_fairness.agreement import (
    ai_vs_pooled_raters_kappa,
    build_inter_rater_evidence,
    cohen_kappa_matrix,
    fleiss_kappa,
    krippendorff_alpha,
    pool_raters,
)
from fmm_fairness.cli import main
from tests.test_multi_class import _make_ai4skin_shaped_df


class TestPoolRaters(unittest.TestCase):
    def test_majority_vote_breaks_ties_to_lowest_class(self) -> None:
        df = pd.DataFrame({"a": [1, 0, 0], "b": [0, 1, 0], "c": [1, 1, 1]})
        pooled = pool_raters(df, ["a", "b", "c"])
        # Row 0: votes 1,0,1 -> majority 1; row 1: 0,1,1 -> 1; row 2: all 0/0/1 -> 0
        np.testing.assert_array_equal(pooled, np.array([1, 1, 0], dtype=int))

    def test_all_missing_returns_sentinel(self) -> None:
        df = pd.DataFrame({"a": [-1], "b": [-1]})
        pooled = pool_raters(df, ["a", "b"])
        self.assertEqual(int(pooled[0]), -1)


class TestCohenKappaMatrix(unittest.TestCase):
    def test_perfect_agreement_is_one(self) -> None:
        df = pd.DataFrame({"r1": [0, 1, 2, 1, 0], "r2": [0, 1, 2, 1, 0]})
        m = cohen_kappa_matrix(df, ["r1", "r2"])
        assert hasattr(m, "matrix")
        self.assertAlmostEqual(m.matrix.loc["r1", "r2"], 1.0, places=10)
        self.assertAlmostEqual(m.matrix.loc["r2", "r1"], 1.0, places=10)
        self.assertAlmostEqual(m.matrix.loc["r1", "r1"], 1.0, places=10)

    def test_missing_ratings_dropped_pairwise(self) -> None:
        df = pd.DataFrame({"r1": [0, 1, -1, 1, 0], "r2": [0, -1, 2, 1, 0]})
        m = cohen_kappa_matrix(df, ["r1", "r2"], missing_value=-1)
        assert hasattr(m, "matrix")
        # only rows 0, 3, 4 are valid for both raters; ratings are 0,1,0 vs 0,1,0
        self.assertAlmostEqual(m.matrix.loc["r1", "r2"], 1.0, places=10)

    def test_stratified_by_returns_dict(self) -> None:
        df = pd.DataFrame(
            {
                "r1": [0, 1, 0, 1],
                "r2": [0, 1, 1, 0],
                "site": ["A", "A", "B", "B"],
            }
        )
        out = cohen_kappa_matrix(df, ["r1", "r2"], stratify_by="site")
        assert isinstance(out, dict)
        self.assertIn("A", out)
        self.assertIn("B", out)

    def test_ai_column_appended(self) -> None:
        df = pd.DataFrame(
            {
                "r1": [0, 1, 2, 1, 0],
                "r2": [0, 1, 2, 1, 1],
                "y_pred": [0, 1, 2, 0, 0],
            }
        )
        m = cohen_kappa_matrix(df, ["r1", "r2"], ai_col="y_pred")
        assert hasattr(m, "matrix")
        self.assertIn("y_pred", m.matrix.index)
        self.assertIn("y_pred", m.matrix.columns)


class TestFleissKappa(unittest.TestCase):
    def test_perfect_agreement_yields_kappa_one(self) -> None:
        df = pd.DataFrame({f"r{i}": [0, 1, 2, 1, 0] for i in range(6)})
        out = fleiss_kappa(df, [f"r{i}" for i in range(6)])
        self.assertAlmostEqual(out.value, 1.0, places=10)
        self.assertEqual(out.n_items, 5)

    def test_textbook_fleiss_example(self) -> None:
        # Fleiss's "diagnostic rating" example: 30 patients, 6 raters, 5 categories.
        # The canonical paper (Fleiss 1971) reports kappa ~0.43 for a synthesised
        # high-agreement subset. Here we use a small constructed case where the
        # math hand-verifies. Five items, four raters, two categories.
        #
        # Item-by-item:
        #   item 1: 4-0  -> p_i = (16 - 4) / (4 * 3) = 1.0
        #   item 2: 3-1  -> p_i = (9 + 1 - 4) / 12 = 6/12 = 0.5
        #   item 3: 2-2  -> p_i = (4 + 4 - 4) / 12 = 4/12 = 1/3
        #   item 4: 4-0  -> p_i = 1.0
        #   item 5: 3-1  -> p_i = 0.5
        # P_bar = (1 + 0.5 + 1/3 + 1 + 0.5) / 5 = 3.333/5 = 0.6667
        # marginals: n_yes = 4+3+2+4+3 = 16, n_no = 4*5 - 16 = 4; p_yes = 16/20=0.8, p_no=0.2
        # P_e = 0.64 + 0.04 = 0.68
        # kappa = (0.6667 - 0.68) / (1 - 0.68) ~= -0.0417
        df = pd.DataFrame(
            {
                "r1": [1, 1, 1, 1, 1],
                "r2": [1, 1, 1, 1, 1],
                "r3": [1, 1, 0, 1, 1],
                "r4": [1, 0, 0, 1, 0],
            }
        )
        out = fleiss_kappa(df, ["r1", "r2", "r3", "r4"])
        self.assertAlmostEqual(out.value, -0.0417, places=3)

    def test_missing_items_excluded(self) -> None:
        df = pd.DataFrame(
            {
                "r1": [0, 1, 0, 1, -1],
                "r2": [0, 1, 0, 1, 0],
                "r3": [0, 1, 0, 1, 0],
            }
        )
        out = fleiss_kappa(df, ["r1", "r2", "r3"])
        self.assertEqual(out.n_items, 4)  # row 4 dropped


class TestKrippendorffAlpha(unittest.TestCase):
    def test_perfect_agreement_yields_alpha_one(self) -> None:
        df = pd.DataFrame({"r1": [0, 1, 2, 1, 0], "r2": [0, 1, 2, 1, 0]})
        out = krippendorff_alpha(df, ["r1", "r2"])
        self.assertAlmostEqual(out.value, 1.0, places=10)

    def test_known_small_case(self) -> None:
        # Two coders, three items: ratings (0,0), (1,0), (1,1).
        # do = pairs disagreement / total pairs.
        # Per item ordered-pair disagreements: item 0: 0, item 1: 2 (1!=0 and 0!=1), item 2: 0
        # total disagreement = 2; total ordered pairs = 2 + 2 + 2 = 6 -> do = 1/3
        # Marginals: 0 appears 3x, 1 appears 3x; n_total = 6.
        # de = (n^2 - sum(n_c^2)) / (n*(n-1)) = (36 - (9+9)) / 30 = 18/30 = 0.6
        # alpha = 1 - (1/3)/0.6 = 1 - 0.5556 = 0.4444
        df = pd.DataFrame({"r1": [0, 1, 1], "r2": [0, 0, 1]})
        out = krippendorff_alpha(df, ["r1", "r2"])
        self.assertAlmostEqual(out.value, 0.4444, places=3)

    def test_missing_ratings_tolerated(self) -> None:
        df = pd.DataFrame({"r1": [0, -1, 1, 0], "r2": [0, 1, 1, 0]})
        out = krippendorff_alpha(df, ["r1", "r2"])
        self.assertEqual(out.n_items, 4)
        # 3 paired items, all agree -> alpha = 1.0
        self.assertAlmostEqual(out.value, 1.0, places=10)


class TestAiVsPooledRatersKappa(unittest.TestCase):
    def test_ai_equal_to_majority_yields_kappa_one(self) -> None:
        df = pd.DataFrame(
            {
                "r1": [0, 1, 2, 0, 1],
                "r2": [0, 1, 2, 0, 1],
                "r3": [0, 1, 2, 0, 1],
                "y_pred": [0, 1, 2, 0, 1],
            }
        )
        out = ai_vs_pooled_raters_kappa(
            df, ["r1", "r2", "r3"], "y_pred", bootstrap_iters=50
        )
        self.assertAlmostEqual(out.value, 1.0, places=10)
        self.assertIsNotNone(out.ci_low)
        self.assertIsNotNone(out.ci_high)

    def test_ai_independent_yields_low_kappa(self) -> None:
        rng = np.random.default_rng(99)
        n = 300
        df = pd.DataFrame(
            {
                "r1": rng.integers(0, 3, size=n),
                "r2": rng.integers(0, 3, size=n),
                "r3": rng.integers(0, 3, size=n),
                "y_pred": rng.integers(0, 3, size=n),
            }
        )
        out = ai_vs_pooled_raters_kappa(
            df, ["r1", "r2", "r3"], "y_pred", bootstrap_iters=100
        )
        self.assertLess(abs(out.value), 0.20)


class TestEvidenceBlock(unittest.TestCase):
    def test_block_has_all_metrics(self) -> None:
        df = _make_ai4skin_shaped_df(rater_columns=True)
        block = build_inter_rater_evidence(
            df,
            rater_cols=[f"doc{i}" for i in range(1, 11)],
            ai_col="y_pred",
            stratify_by="site",
            bootstrap_iters=50,
        )
        for key in (
            "fleiss_kappa",
            "krippendorff_alpha",
            "cohen_kappa_matrix",
            "ai_vs_pooled_raters_kappa",
            "cohen_kappa_matrix_by_stratum",
        ):
            self.assertIn(key, block)
        self.assertEqual(block["stratified_by"], "site")
        # AI is a reasonable approximation to ground truth in the fixture;
        # kappa should land well above chance.
        self.assertGreater(block["ai_vs_pooled_raters_kappa"]["value"], 0.30)


class TestCliEndToEndWithRaterCols(unittest.TestCase):
    def test_evaluate_emits_inter_rater_block(self) -> None:
        df = _make_ai4skin_shaped_df(rater_columns=True)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "preds.csv"
            df.to_csv(csv_path, index=False)
            out_dir = Path(tmp) / "report"
            rc = main(
                [
                    "evaluate",
                    str(csv_path),
                    "--protected-attrs",
                    "site",
                    "--site-attribute",
                    "site",
                    "--num-classes",
                    "6",
                    "--rater-cols",
                    ",".join(f"doc{i}" for i in range(1, 11)),
                    "--output",
                    str(out_dir),
                ]
            )
            self.assertEqual(rc, 0)
            evidence = json.loads(
                (out_dir / "fairness-evidence.json").read_text(encoding="utf-8")
            )
            self.assertIn("inter_rater_agreement", evidence)
            block = evidence["inter_rater_agreement"]
            self.assertEqual(len(block["rater_columns"]), 10)
            self.assertEqual(block["ai_column"], "y_pred")
            self.assertIn("cohen_kappa_matrix", block)
            self.assertIn("ai_vs_pooled_raters_kappa", block)
            # Markdown report contains the matrix block.
            md = (out_dir / "fairness-report.md").read_text(encoding="utf-8")
            self.assertIn("Inter-rater agreement", md)
            self.assertIn("doc1", md)
            self.assertIn("Fleiss kappa", md)

    def test_evaluate_rejects_missing_rater_columns(self) -> None:
        df = _make_ai4skin_shaped_df()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "preds.csv"
            df.to_csv(csv_path, index=False)
            rc = main(
                [
                    "evaluate",
                    str(csv_path),
                    "--protected-attrs",
                    "site",
                    "--num-classes",
                    "6",
                    "--rater-cols",
                    "doc1,doc2,doc3",
                    "--output",
                    str(Path(tmp) / "out"),
                ]
            )
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
