# Changelog

All notable changes to this project will be documented here.

## v0.2.0a5 -- 2026-05-24 (AI4SkIN golden replication example — roadmap S5)

- New `examples/ai4skin-replication/` directory bundling a CLI-shape replication of the TFG inter-site fairness headline:
  - `confusion_matrices.json` ships the published HUSC (n=116) and HCUV (n=41) ABMIL+UNI confusion matrices, extracted cell-by-cell from the thesis figures (`bcmHUCV.png`, `blue_confusion_matrix_HCUV.png`).
  - `build_dataset.py` deterministically expands the matrices into a row-per-sample `predictions.csv` with `y_score_0..y_score_5` synthesised to peak at `y_pred`, plus a `raters.csv` of 10 synthetic raters whose disagreements follow the model's own confusion distribution (parameterised to land in the published-plausible AI-vs-pooled κ band 0.70-0.90).
  - `replicate.ipynb` runs the CLI end-to-end, prints the headline numbers, and asserts the weighted F1 gap reproduces TFG Table 6 within ±0.005.
  - `README.md` documents transparently that the thesis cites two distinct weighted F1 figures: the abstract value (0.241 gap, 0.690/0.931) and the per-cell confusion matrix value (0.166 gap, 0.757/0.922). This replication targets the second because it is the one a third party can audit against disclosed artefacts.
- New `tests/test_ai4skin_replication.py` (5 tests) runs the same CLI invocation as the notebook and gates: predictions CSV has exactly 157 rows, weighted F1 gap is 0.1657 ± 0.005 with per-group values HUSC=0.9224 and HCUV=0.7567 (both ±0.005), permutation p < 0.05, MDE@80% power is finite and < 0.30, and AI-vs-pooled κ lands in [0.70, 0.90].
- No source-code changes: this slice exercises the v0.2.0a4 API surface on a published-real fixture, closing the "does the CLI line up with the TFG?" question for v0.2 GA.

## v0.2.0a4 -- 2026-05-21 (statistical rigour — roadmap S4)

- New module `fmm_fairness.statistics` providing the v0.2-roadmap inference layer:
  - `bca_bootstrap_gap_ci(...)` — bias-corrected and accelerated (Efron 1987) bootstrap CI over a per-group statistic's max-min gap. Falls back to percentile when BCa endpoints are degenerate.
  - `percentile_bootstrap_gap_ci(...)` — preserved v0.1-compatible interval, available behind a flag.
  - `permutation_test_gap_pvalue(...)` — two-sided label-shuffle permutation test for `H0 = no gap` with the standard `(n_extreme + 1) / (n_iters + 1)` plug-in.
  - `minimum_detectable_effect(bootstrap_se, alpha, power)` — `(z_{1 - alpha/2} + z_{power}) * SE_gap` MDE summary; uses the bootstrap SE so it works without a closed-form variance.
  - `cohens_d(...)` and `odds_ratio_binary(...)` effect-size helpers.
  - `gap_inference(...)` orchestrator returning a `GapInference` bundle (CI + p-value + MDE) for one gap statistic.
- `weighted_f1_gap` and `macro_f1_gap` now default to BCa and accept `--bootstrap-method {bca, percentile}`, `--bootstrap-iters`, `--permutation-iters`, `--alpha`, `--power` via the CLI and via the Python API. Their results carry `bootstrap_method`, `bootstrap_se`, `permutation_p_value`, `permutation_iters`, `minimum_detectable_effect`, `alpha`, and `power` fields, all surfaced in the JSON evidence pack.
- 11 new tests: BCa vs percentile parity on the SE, permutation test recovers `p > 0.05` on a no-gap fixture and `p < 0.01` on a clear-gap fixture, MDE monotone-in-SE, BCa metadata round-trips through `FairnessResult.to_dict()`, Cohen's d and odds ratio known-value cases.
- `scipy>=1.10` added to `dependencies` (used for `scipy.stats.norm` in BCa endpoint adjustment).
- New `docs/statistical-methodology.md` covering BCa derivation, permutation-test caveats (conditioning, multiple comparisons), the MDE-not-observed-power justification (Hoenig & Heisey 2001), and the AI Act Art. 9 / Art. 15 mapping.

## v0.2.0a3 -- 2026-05-21 (foundation-model comparison mode — roadmap S3)

- New CLI subcommand `fmm-fairness compare <csv1> <csv2> ... --labels uni,conch,plip,...` that ranks foundation-model candidates on a joint accuracy + fairness Pareto frontier and recommends one under an EU AI Act Art. 9 residual-risk heuristic.
- New module `fmm_fairness.comparison`:
  - `compare_models(dfs, labels, ...)` returns a `ComparisonResult` carrying per-model `ModelEvaluation`s (overall weighted/macro F1, inter-site weighted/macro F1 gap, SaMD fairness score), the Pareto-frontier and dominated label sets, and the recommended candidate with rationale.
  - `_pareto_frontier(points)` computes non-dominated indices over (perf, gap) point pairs with the convention higher-perf and lower-gap are both better.
  - `_recommend_from_frontier(...)` picks the frontier candidate with the highest overall weighted F1 subject to `--fairness-floor`; falls back to the fairest frontier candidate if no candidate meets the floor, with an explicit rationale flag.
- Comparison evidence pack: `comparison-report.md` + `comparison-evidence.json` + `audit.sha256`, parallel to the `evaluate` pack shape.
- 12 new tests: Pareto-frontier unit tests (dominance, ties, identity), three-candidate `uni/conch/plip` synthetic comparison with hand-verified relative ordering, tight-floor fallback path, CLI end-to-end test, mismatched-label validation.
- New `docs/foundation-model-comparison.md` covering frontier semantics, the AI Act Art. 9 framing, a worked three-candidate example, limitations, and references.

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
