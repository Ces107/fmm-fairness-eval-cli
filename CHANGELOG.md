# Changelog

All notable changes to this project will be documented here.

## v0.2.0a2 -- 2026-05-21 (inter-rater agreement — roadmap S2)

- New module `fmm_fairness.agreement` with the four canonical inter-rater agreement statistics:
  - `cohen_kappa_matrix(df, rater_cols, ai_col=None, stratify_by=None)` — pairwise Cohen kappa across raters and (optionally) the AI column. Pairwise missing-value handling (sentinel -1 by default). Optional per-stratum matrix when a stratifying column is given.
  - `fleiss_kappa(df, rater_cols)` — single global agreement scalar for a fixed-N rater panel. Items with any missing rating are excluded.
  - `krippendorff_alpha(df, rater_cols)` — nominal-scale alpha; tolerates missing ratings via per-item disagreement counting.
  - `ai_vs_pooled_raters_kappa(df, rater_cols, ai_col)` — Cohen kappa between AI predictions and the per-item majority vote of the raters, with percentile-bootstrap 95% CI. The headline SaMD validation scalar.
- CLI: new `--rater-cols doc1,doc2,...,doc10` flag plus `--rater-missing-value` override.
- Evidence pack gains an `inter_rater_agreement` block when rater columns are declared, with the full statistics suite plus a per-site stratified Cohen kappa matrix when the site attribute is present. Markdown report renders the global matrix inline.
- AI Act manifest mode (`--manifest-mode ai-act`) cross-cites the agreement statistics to Art. 14 (human oversight).
- 17 new tests covering perfect-agreement identity cases, a hand-verified Fleiss kappa textbook example, a small Krippendorff alpha case with hand-verified value, missing-rating tolerance, per-site stratification, AI-vs-pooled kappa with independent random fixture, and full CLI end-to-end with the AI4SkIN-shaped fixture extended with 10 synthetic rater columns.
- New `docs/inter-rater-agreement.md` covering all four statistics, the Landis-Koch interpretation cut-offs, pooled-vs-pairwise rationale, tie-break choice, regulatory mapping, and references.

## v0.2.0a1 -- 2026-05-21 (multi-class data model — roadmap S1)

- Multi-class data model: y_true/y_pred in {0..K-1}; scores either binary `y_score` or per-class `y_score_0..y_score_{K-1}`. K auto-detected from the score columns; `--num-classes K` flag overrides with a warning on mismatch.
- New fairness metrics for K >= 2:
  - `weighted_f1_gap(df, attribute)` — support-weighted F1 max-min across groups; the TFG headline disparity.
  - `macro_f1_gap(df, attribute)` — equal-class-weight F1 gap.
  - `per_class_f1_gap(df, attribute)` — per-class breakdown of which class drives the inter-group disparity; returns full length-K per-group vectors and a length-K gap vector; headline `gap` is the worst-class entry.
  - `multi_class_auc_gap(df, attribute)` — max-min OVR macro AUC across groups (binary AUC for K=2).
- `inter_site_auc_variance` now dispatches on K: bit-identical numerics for K=2, OVR macro AUC for K>2.
- `samd_fairness_score` extended with the `F1_SITE` component (weighted F1 gap at the site attribute) at weight 0.35 by default. Five-component formula:
  ```
  raw = 0.35 * F1_SITE + 0.20 * SITE + 0.20 * EO + 0.10 * DP + 0.15 * CAL
  ```
  Under K>2 the binary-only terms (EO/DP/CAL) collapse to zero and `{w_f1_site, w_site}` renormalise to sum to 1.0. The output carries both `weights` (effective) and `weights_declared` (inputs) for audit symmetry.
- Evidence pack is K-aware: binary CSVs receive the EO/DP/CAL block alongside the F1 family; multi-class CSVs receive the F1 family only (binary-only criteria require per-class operating thresholds the CLI does not yet ingest — parked for roadmap S6).
- Markdown report surfaces K, per-class F1 vectors per group, and per-class gap vectors.
- AI Act manifest mode (`--manifest-mode ai-act`) picks K-aware Art. 10 metric list.
- 27 new tests across `tests/test_multi_class.py` and `tests/test_cli_multi_class.py`; total suite is 41 tests, ruff + mypy --strict clean.
- Docs: new `docs/multi-class-metrics.md`; `docs/samd-fairness-score.md` updated with v0.2 formula, K-aware weight renormalisation, and worked multi-class example reproducing the TFG-shape composite.

Backward compatibility for binary inputs: per-metric numerics for `equal_opportunity_gap`, `demographic_parity_gap`, `calibration_gap`, and `inter_site_auc_variance` are unchanged from v0.1. The `samd_fairness_score` composite changes numerically because of the new `F1_SITE` component; the old behaviour can be approximated via `weights={"f1_site": 0.0, "site": 0.4, "eo": 0.3, "dp": 0.15, "cal": 0.15}`.

## v0.1.0 -- 2026-05-15 (initial release)

- CLI `fmm-fairness evaluate` computes Equal Opportunity, Demographic Parity, Calibration, Inter-site AUC variance, and the composite `samd_fairness_score`.
- Emits human-readable `fairness-report.md` plus machine-readable `fairness-evidence.json`.
- SHA-256 audit chain over each report (file `audit.sha256` companion).
- `--manifest-mode ai-act` toggle cross-cites the evidence to EU AI Act Art. 9 + Art. 10.
- 14 unittest cases passing under Python 3.10/3.11/3.12.
- Documented `samd_fairness_score` composite at `docs/samd-fairness-score.md`.
- Example workflow at `docs/example-workflow.md` walks through synthetic dermatology-AI predictions to evidence-pack.
- Research-artifact framing accompanying the author's TFG on fairness-aware SaMD (https://riunet.upv.es/handle/10251/226903).

## Roadmap (v0.2 candidates, dependent on signal)

- Pre-built protected-attribute discovery heuristics.
- Bootstrap-based confidence intervals on `samd_fairness_score`.
- Calibration-by-subgroup with Hosmer-Lemeshow goodness-of-fit.
- Integration with MONAI Bundle outputs.
- Hosted CI mode (€99-149/mo) running fairness eval per PR.
