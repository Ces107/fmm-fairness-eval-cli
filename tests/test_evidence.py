"""Tests for fmm_fairness.evidence JSON shape and audit chain."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fmm_fairness.evidence import (
    EVIDENCE_FILENAME,
    EvaluationConfig,
    build_evidence,
    write_evidence_pack,
)


def _fixture_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    sites = rng.choice(["A", "B"], size=n)
    y_true = rng.integers(0, 2, size=n)
    y_score = np.where(y_true == 1, rng.beta(7, 2, size=n), rng.beta(2, 7, size=n))
    y_pred = (y_score >= 0.5).astype(int)
    sex = rng.choice(["F", "M"], size=n)
    return pd.DataFrame({
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
        "y_score": y_score,
        "site": sites,
        "sex": sex,
    })


class TestEvidenceShape(unittest.TestCase):
    def test_evidence_has_required_keys(self) -> None:
        df = _fixture_df()
        cfg = EvaluationConfig(
            predictions_path="fixture.csv",
            protected_attrs=["site", "sex"],
            site_attribute="site",
            timestamp_iso="2026-05-14T00:00:00Z",
        )
        e = build_evidence(df, cfg)
        for k in ("tool", "tool_version", "evidence_schema_version",
                  "generated_at_utc", "n_samples",
                  "protected_attributes_declared", "per_attribute_metrics",
                  "samd_fairness_score", "inter_site"):
            self.assertIn(k, e)
        self.assertEqual(e["tool"], "fmm-fairness-eval")
        self.assertIn("site", e["per_attribute_metrics"])
        self.assertIn("sex", e["per_attribute_metrics"])
        self.assertIsInstance(e["evidence_schema_version"], str)
        self.assertRegex(
            e["evidence_schema_version"],
            r"^\d{4}\.\d{2}\.\d{2}$",
            "evidence_schema_version should be a date-keyed string YYYY.MM.DD",
        )

    def test_evidence_pack_is_strict_json_serialisable(self) -> None:
        """TD-024: written pack must parse with strict JSON (no NaN/Inf literals)."""
        df = _fixture_df()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EvaluationConfig(
                predictions_path="fixture.csv",
                protected_attrs=["site", "sex"],
                site_attribute="site",
                bootstrap_iters=100,
                output_dir=tmp,
            )
            write_evidence_pack(df, cfg)
            raw = (Path(tmp) / EVIDENCE_FILENAME).read_text(encoding="utf-8")
            # Standard JSON does not accept the literals NaN / Infinity / -Infinity.
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            # Strict reparse (Python's json.loads with no flags is RFC 8259 strict
            # and will reject NaN/Inf literals); also try the explicit allow_nan
            # roundtrip on the dumped string.
            parsed = json.loads(raw)
            json.dumps(parsed, allow_nan=False)

    def test_ai_act_mode_adds_regulatory_mapping(self) -> None:
        df = _fixture_df()
        cfg = EvaluationConfig(
            predictions_path="fixture.csv",
            protected_attrs=["site", "sex"],
            manifest_mode="ai-act",
        )
        e = build_evidence(df, cfg)
        self.assertIn("regulatory_mapping", e)
        self.assertEqual(e["regulatory_mapping"]["framework"],
                         "EU AI Act (Regulation 2024/1689)")
        arts = [a["article"] for a in e["regulatory_mapping"]["articles"]]
        self.assertIn("Art. 10", arts)
        self.assertIn("Art. 9", arts)

    def test_write_evidence_pack_creates_three_files_with_consistent_sha(self) -> None:
        df = _fixture_df()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EvaluationConfig(
                predictions_path="fixture.csv",
                protected_attrs=["site", "sex"],
                output_dir=tmp,
                timestamp_iso="2026-05-14T00:00:00Z",
            )
            result = write_evidence_pack(df, cfg)
            for k in ("evidence_json", "report_md", "audit_sha256"):
                self.assertTrue(Path(result[k]).exists())
            # Round-trip the JSON
            obj = json.loads(Path(result["evidence_json"]).read_text(encoding="utf-8"))
            self.assertEqual(obj["tool"], "fmm-fairness-eval")
            # Audit file mentions both SHAs
            audit_text = Path(result["audit_sha256"]).read_text(encoding="utf-8")
            self.assertIn(result["evidence_json_sha256"], audit_text)
            self.assertIn(result["report_md_sha256"], audit_text)


if __name__ == "__main__":
    unittest.main()
