"""Tests for the foundation-model comparison subsystem.

Three synthetic candidates are constructed with deliberately different
(accuracy, inter-site fairness) profiles. The fixture is small but the
relative ordering is verified analytically:

- ``uni``    : moderately accurate, low inter-site disparity.
- ``conch``  : highly accurate, moderate inter-site disparity.
- ``plip``   : poorly accurate, high inter-site disparity.

Expected Pareto frontier: ``{uni, conch}``. ``plip`` is dominated by
both. Default recommendation: ``conch`` if its inter-site gap is at or
below the fairness floor, else ``uni`` (the fairest frontier candidate).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fmm_fairness.cli import main
from fmm_fairness.comparison import (
    ComparisonResult,
    _pareto_frontier,
    compare_models,
)
from tests.test_multi_class import (
    NUM_CLASSES,
    SCORE_COL_PREFIX,
    _softmax_rows,
)


def _make_candidate(
    seed: int,
    *,
    n_per_site: int = 300,
    acc_strong: float = 0.90,
    acc_weak: float = 0.85,
    site_b_strong: float = 0.90,
    site_b_weak: float = 0.85,
) -> pd.DataFrame:
    """Build a 6-class 2-site predictions DataFrame with tunable per-class accuracy."""
    rng = np.random.default_rng(seed)
    K = NUM_CLASSES
    rows = []
    prevalence = np.array([0.30, 0.25, 0.18, 0.12, 0.10, 0.05])
    for site, acc_s, acc_w in (
        ("HCUV", acc_strong, acc_weak),
        ("HUSC", site_b_strong, site_b_weak),
    ):
        n = n_per_site
        y_true = rng.choice(np.arange(K), size=n, p=prevalence)
        y_pred = np.empty(n, dtype=int)
        for i, cls in enumerate(y_true):
            weak = cls >= 3
            accuracy = acc_w if weak else acc_s
            if rng.random() < accuracy:
                y_pred[i] = cls
            else:
                other = rng.integers(0, K - 1)
                if other >= cls:
                    other += 1
                y_pred[i] = other
        logits = rng.normal(0.0, 0.4, size=(n, K))
        logits[np.arange(n), y_pred] += 3.5
        probs = _softmax_rows(logits)
        site_rows = {f"{SCORE_COL_PREFIX}{k}": probs[:, k] for k in range(K)}
        site_rows["y_true"] = y_true.astype(int)
        site_rows["y_pred"] = y_pred.astype(int)
        site_rows["site"] = np.full(n, site)
        rows.append(pd.DataFrame(site_rows))
    return pd.concat(rows, ignore_index=True)


def _make_three_candidates() -> dict[str, pd.DataFrame]:
    return {
        # uni: moderate accuracy, low inter-site gap (HCUV ~ HUSC)
        "uni": _make_candidate(
            seed=1, acc_strong=0.78, acc_weak=0.72, site_b_strong=0.78, site_b_weak=0.72
        ),
        # conch: high accuracy, moderate inter-site gap
        "conch": _make_candidate(
            seed=2, acc_strong=0.95, acc_weak=0.90, site_b_strong=0.85, site_b_weak=0.65
        ),
        # plip: poor accuracy, high inter-site gap (dominated by both)
        "plip": _make_candidate(
            seed=3, acc_strong=0.60, acc_weak=0.55, site_b_strong=0.45, site_b_weak=0.25
        ),
    }


class TestParetoFrontier(unittest.TestCase):
    def test_single_point_is_on_frontier(self) -> None:
        self.assertEqual(_pareto_frontier([(0.5, 0.5)]), [0])

    def test_two_dominated_one_dominant(self) -> None:
        # Point 2 (perf=0.9, gap=0.05) dominates both others.
        points = [(0.7, 0.15), (0.6, 0.20), (0.9, 0.05)]
        self.assertEqual(_pareto_frontier(points), [2])

    def test_no_dominance_relation_keeps_all(self) -> None:
        # Three points on a true frontier — none dominates another.
        points = [(0.6, 0.05), (0.7, 0.10), (0.8, 0.15)]
        self.assertEqual(sorted(_pareto_frontier(points)), [0, 1, 2])

    def test_ties_break_to_keep_both(self) -> None:
        # Two identical points: neither dominates the other (no strict inequality).
        points = [(0.7, 0.1), (0.7, 0.1)]
        self.assertEqual(sorted(_pareto_frontier(points)), [0, 1])


class TestCompareModels(unittest.TestCase):
    def test_three_candidate_frontier(self) -> None:
        candidates = _make_three_candidates()
        result = compare_models(
            list(candidates.values()),
            list(candidates.keys()),
            site_attribute="site",
        )
        self.assertIsInstance(result, ComparisonResult)
        # plip must be dominated by both uni and conch
        self.assertIn("plip", result.pareto_dominated_labels)
        self.assertIn("conch", result.pareto_frontier_labels)
        # uni may or may not be on the frontier depending on relative noise,
        # but the dominated set should NOT include both uni and conch.
        self.assertNotEqual(
            set(result.pareto_dominated_labels), {"uni", "conch", "plip"}
        )

    def test_recommendation_present(self) -> None:
        candidates = _make_three_candidates()
        result = compare_models(
            list(candidates.values()),
            list(candidates.keys()),
            site_attribute="site",
            fairness_floor=0.50,  # loose floor; almost any candidate qualifies
        )
        self.assertIsNotNone(result.recommended_label)
        self.assertIn(result.recommended_label, candidates.keys())
        self.assertIn("Art. 9", result.recommendation_rationale or "")

    def test_tight_fairness_floor_triggers_fallback(self) -> None:
        candidates = _make_three_candidates()
        result = compare_models(
            list(candidates.values()),
            list(candidates.keys()),
            site_attribute="site",
            fairness_floor=0.0,  # impossibly tight floor
        )
        rationale = result.recommendation_rationale or ""
        # The fallback branch must announce that no model met the floor.
        self.assertIn("No frontier model meets the configured fairness floor", rationale)

    def test_duplicate_labels_rejected(self) -> None:
        df = _make_candidate(seed=10)
        with self.assertRaises(ValueError):
            compare_models([df, df], ["same", "same"], site_attribute="site")

    def test_single_candidate_rejected(self) -> None:
        df = _make_candidate(seed=11)
        with self.assertRaises(ValueError):
            compare_models([df], ["uni"], site_attribute="site")

    def test_mismatched_label_count_rejected(self) -> None:
        df1 = _make_candidate(seed=12)
        df2 = _make_candidate(seed=13)
        with self.assertRaises(ValueError):
            compare_models([df1, df2], ["uni"], site_attribute="site")


class TestCliCompareEndToEnd(unittest.TestCase):
    def test_compare_writes_pack_and_returns_zero(self) -> None:
        candidates = _make_three_candidates()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for label, df in candidates.items():
                p = Path(tmp) / f"{label}.csv"
                df.to_csv(p, index=False)
                paths.append(str(p))
            out_dir = Path(tmp) / "comparison"
            rc = main(
                [
                    "compare",
                    *paths,
                    "--labels",
                    ",".join(candidates.keys()),
                    "--protected-attrs",
                    "site",
                    "--site-attribute",
                    "site",
                    "--num-classes",
                    "6",
                    "--output",
                    str(out_dir),
                ]
            )
            self.assertEqual(rc, 0)
            evidence = json.loads(
                (out_dir / "comparison-evidence.json").read_text(encoding="utf-8")
            )
            self.assertEqual(evidence["command"], "compare")
            self.assertEqual(evidence["labels"], list(candidates.keys()))
            self.assertIn("result", evidence)
            self.assertEqual(len(evidence["result"]["models"]), 3)
            md = (out_dir / "comparison-report.md").read_text(encoding="utf-8")
            self.assertIn("Foundation-model comparison report", md)
            self.assertIn("Pareto frontier", md)
            for label in candidates:
                self.assertIn(label, md)

    def test_compare_rejects_mismatched_labels(self) -> None:
        candidates = _make_three_candidates()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for label, df in candidates.items():
                p = Path(tmp) / f"{label}.csv"
                df.to_csv(p, index=False)
                paths.append(str(p))
            rc = main(
                [
                    "compare",
                    *paths,
                    "--labels",
                    "uni,conch",  # only 2 labels for 3 CSVs
                    "--protected-attrs",
                    "site",
                    "--output",
                    str(Path(tmp) / "out"),
                ]
            )
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
