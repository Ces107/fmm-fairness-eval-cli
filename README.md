# fmm-fairness-eval

> SaMD-specific fairness evaluation CLI for foundation-model medical AI. Emits AI Act Art. 10 / Art. 9 evidence artifacts.

## Pricing

- **CLI (Free)**. OSS Python CLI, MIT-licensed, fairness metrics + audit chain.
- **Hosted CI (€99/mo)**. Hosted runner on customer prediction CSVs, monthly evidence pack.
- **Consulting (€60-100/hour)**. Fairness-evidence drafting, AI Act Art. 9 / Art. 10 dossier review, Calendly booking.

Paid tiers open early-access via email; the CLI itself is free under MIT and `pip install --pre fmm-fairness-eval` works today. See `pricing.md` for tier details and how to request early access.


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`fmm-fairness-eval` (`fmm-fairness` on the command line) is a small, focused CLI that takes a predictions CSV from any SaMD or SaMD-adjacent medical-AI model and produces a regulator-friendly fairness evidence pack: a Markdown report, a machine-readable JSON pack, and a SHA-256 audit chain. It is built around the failure mode regulators actually care about: inter-hospital and inter-site bias. The output drops straight into an EU AI Act Art. 10 / Art. 9 dossier.

---

## Why this exists

Modern medical-AI systems are increasingly built on **foundation-model embeddings** (CONCH for histopathology, DINOv2 for general radiology, RadFM-style models for multi-modal radiology) plus a small downstream classifier. The dominant failure mode is no longer "the model is biased against women" or "the model misses dark skin tones" in isolation. It is **inter-hospital generalization collapse**: the model that scores F1=0.89 on one cohort drops to F1=0.70 on the cohort across the river, and the gap is largest in subgroups the training data under-represented.

The author's TFG (Universitat Politècnica de València, 2024) measured exactly this on dermatology AI using CONCH embeddings and multiple-instance learning over the AI4SkIN cohort: weighted F1 = 0.89, but an inter-hospital fairness gap of 0.19 between sites. That is not a corner case. It is the modal failure mode for any SaMD that crosses a hospital network boundary.

Existing fairness libraries (FairLearn, AIF360, Holistic AI, Microsoft Responsible AI Toolbox) are general-purpose ML fairness tools. None of them ship a SaMD-specific evaluation pipeline that:

- Treats `site` / `hospital` as a first-class protected attribute distinct from individual demographics.
- Emits AI Act Art. 10 / Art. 9 cross-cited evidence by default.
- Defines a composite SaMD fairness score whose weighting reflects how regulators actually prioritize bias categories.
- Ships a SHA-256 audit chain so the evidence pack is tamper-evident the moment it leaves your pipeline.

This tool fills that gap. Nothing more, nothing less.

Citation for the underlying TFG work: César Pereiro, _Foundation-model-based fairness evaluation in dermatology classification using the AI4SkIN dataset_, Universitat Politècnica de València, 2024. https://riunet.upv.es/handle/10251/226903

---

## Install

The current feature set (BCa bootstrap, permutation tests, intersectional
analysis, multi-class, AI Act full dossier) is in the `0.2.0a` pre-release
line. Install it with `--pre`:

```bash
pip install --pre fmm-fairness-eval
```

A bare `pip install fmm-fairness-eval` (without `--pre`) currently resolves
to the older `0.1.0` stable, which lacks most of the capability documented
below. Use `--pre` until `0.2.0` is promoted to stable.

Or from source:

```bash
git clone https://github.com/Ces107/fmm-fairness-eval-cli
cd fmm-fairness-eval-cli
pip install -e .
```

Requires Python 3.10+, numpy ≥ 1.24, pandas ≥ 2.0, scikit-learn ≥ 1.3. No GPU dependency required.

---

## What it does

### 1. Run an evaluation

```bash
fmm-fairness evaluate predictions.csv \
    --protected-attrs site,sex,age_bucket \
    --site-attribute site \
    --output fairness-report/
```

`predictions.csv` must contain these columns, where the score-column shape depends on the number of classes `K` your model produces.

**Binary (K = 2):**

| column     | type             | meaning                                  |
|------------|------------------|------------------------------------------|
| `y_true`   | int ∈ {0, 1}     | Ground-truth label                       |
| `y_pred`   | int ∈ {0, 1}     | Thresholded prediction                   |
| `y_score`  | float ∈ [0, 1]   | Raw model probability `P(class = 1)`     |
| (declared) | str              | One column per `--protected-attrs` value |

**Multi-class (K ≥ 2):**

| column            | type              | meaning                                  |
|-------------------|-------------------|------------------------------------------|
| `y_true`          | int ∈ {0..K-1}    | Ground-truth label                       |
| `y_pred`          | int ∈ {0..K-1}    | Argmax prediction                        |
| `y_score_0`       | float ∈ [0, 1]    | `P(class = 0)`                           |
| `y_score_1`       | float ∈ [0, 1]    | `P(class = 1)`                           |
| ...               | ...               | ...                                      |
| `y_score_{K-1}`   | float ∈ [0, 1]    | `P(class = K-1)` (rows sum to ~1.0)      |
| (declared)        | str               | One column per `--protected-attrs` value |

The CLI auto-detects `K` from the score columns. Pass `--num-classes K` to override or to assert the expected value; a mismatch with the detected shape triggers a warning.

For multi-class inputs the evidence pack carries the F1-family fairness metrics (`weighted_f1_gap`, `macro_f1_gap`, `per_class_f1_gap`, `multi_class_auc_gap`); see [`docs/multi-class-metrics.md`](docs/multi-class-metrics.md) for definitions. For binary inputs the v0.1 metrics (`equal_opportunity_gap`, `demographic_parity_gap`, `calibration_gap`) are retained.

### 2. Read the output

The CLI produces three files in `fairness-report/`:

- `fairness-report.md`: human-readable, regulator-friendly summary.
- `fairness-evidence.json`: machine-readable evidence pack (stable schema, sorted keys for deterministic SHA).
- `audit.sha256`: SHA-256 of the above two files. Pin in your QMS / change-control record.

### 3. Cross-cite to the AI Act

```bash
fmm-fairness evaluate predictions.csv \
    --protected-attrs site,sex,age_bucket \
    --manifest-mode ai-act \
    --output fairness-report/
```

In `ai-act` mode the JSON pack gains a `regulatory_mapping` block that cross-cites each metric to the EU AI Act article it evidences:

- **Art. 9 (Risk management system)** ↔ `samd_fairness_score`, `inter_site_auc_variance`.
- **Art. 10 (Data and data governance)** ↔ `equal_opportunity_gap`, `demographic_parity_gap`, `calibration_gap`. Evidences Art. 10(2)(f-g) examination of biases and shortcomings.
- **Art. 15 (Accuracy, robustness)** ↔ `inter_site_auc_variance`. Evidences generalization claims.

---

## Metrics computed

| Metric                       | Formula (short)                              | When it matters                                       |
|------------------------------|----------------------------------------------|-------------------------------------------------------|
| `equal_opportunity_gap`      | max-min TPR across groups (Hardt et al. 2016) | Under-diagnosis disparity (Pierson et al. 2021)       |
| `demographic_parity_gap`     | max-min P(ŷ=1) across groups                  | Selection-rate disparity                              |
| `calibration_gap`            | max-min ECE across groups                     | Score-trust differs by subgroup                       |
| `inter_site_auc_variance`    | Var(AUC) across sites                         | Inter-hospital generalization risk (the SaMD failure mode) |
| `samd_fairness_score`        | composite ∈ [0,1] (see `docs/samd-fairness-score.md`) | Single-number summary for QMS dashboards    |

All gap metrics ship with **percentile bootstrap 95% CIs** computed over a stratified resample.

The composite `samd_fairness_score` is defined explicitly with documented weights and a sensitivity analysis in [`docs/samd-fairness-score.md`](docs/samd-fairness-score.md). It is not a black box and is not an FDA-blessed metric. It is a transparent aggregate the operator can defend, override, or replace.

---

## Scientific context

- **CONCH** (Lu et al. 2024) is the visual-language pathology foundation model used in the underlying TFG work. Lu, M. Y. et al. "A visual-language foundation model for computational pathology." *Nature Medicine* 30, 863–874 (2024). [doi:10.1038/s41591-024-02856-4](https://doi.org/10.1038/s41591-024-02856-4)
- **AI4SkIN** is the multi-hospital dermatopathology dataset (Spain, multi-site) on which the TFG measured the 0.19 inter-hospital gap.
- **Under-diagnosis bias on chest X-rays** (Seyyed-Kalantari et al. 2021, *Nat. Med.* 27, 2176-2182) is the canonical demonstration that single-site fairness audits miss the dominant failure mode.
- **Pain disparity reduction** (Pierson et al. 2021, *Nat. Med.* 27, 136-140) demonstrates the inverse. Algorithmic predictions can outperform human-graded severity in capturing real disparities, motivating better measurement, not less.
- **Ethical implementation** (Char, Shah, Magnus 2018, *NEJM* 378, 981-983) sets the still-canonical framing for healthcare-ML ethics.

A one-paragraph literature pointer for the formal definitions: the equal-opportunity criterion is Hardt, Price, Srebro (NeurIPS 2016); the demographic-parity definition follows Dwork et al. (ITCS 2012); calibration-by-group follows Pleiss et al. (NeurIPS 2017). The composite weighting is justified from FDA GMLP guidance (2021, updated 2024 IMDRF GMLP) + EU AI Act Art. 10 prioritisation of multi-site data governance.

---

## What it does NOT do

- **Not** a model-training framework. Bring your own predictions.
- **Not** a foundation-model serving stack. Embeddings are outside scope.
- **Not** auto-detection of protected attributes. You must declare them. Silent attribute inference is itself a bias risk.
- **Not** a certification. A fairness evaluation is evidence; certification is a regulatory process this tool helps you prepare for.
- **Not** an explainability tool. It surfaces *where* bias lives, not *why*.

---

## Honest scientific caveats (read before quoting numbers)

1. **Threshold sensitivity.** `equal_opportunity_gap` and `demographic_parity_gap` both depend on the operating threshold used to produce `y_pred`. Re-run the evaluation at any threshold you would actually deploy at.
2. **Small-sample bootstrap.** Percentile bootstrap is approximate for small groups; for n < 50 prefer the BCa interval or treat CIs as exploratory. Groups with n < `min_group_n` (default 20) are excluded with a warning rather than silently producing a near-zero gap.
3. **Prevalence confound.** Inter-site bias is frequently confounded with prevalence shift. A site that sees twice the disease prevalence will have different TPR even from a perfectly fair model. The tool reports both per-site AUC (less prevalence-sensitive) and per-site rates; interpret jointly.
4. **Composite score is opinionated.** The default weights (`w_site=0.4, w_eo=0.3, w_dp=0.15, w_cal=0.15`) reflect this author's read of regulatory priority. Override with `weights=` in the Python API or treat the components separately. The single number is for dashboards; the components are for decisions.
5. **No causal inference.** A measured gap does not identify the mechanism. Combine with subgroup analysis, training-set provenance audit, and (where possible) prospective evaluation.

---

## Pricing

- **CLI**: MIT, free, forever.
- **Hosted "fairness CI" (Phase 2, not yet shipped)**: planned at €99/month for teams that want every commit to a model repo to fire an evaluation against a frozen multi-site cohort and post the evidence pack as a CI artifact. Mailing list opens at validation green-light.
- **Consulting**: the author is available for SaMD fairness review / AI Act Art. 10 evidence-pack design at €60-100/hour. Contact via the linked GitHub profile; introductions through the academic-DM channel are welcome.

---

## Citing

If you use this tool in published research:

```bibtex
@software{pereiro2026fmmfairness,
  author       = {Pereiro, C{\'e}sar},
  title        = {{fmm-fairness-eval}: SaMD-specific fairness evaluation for foundation-model medical AI},
  year         = {2026},
  url          = {https://github.com/Ces107/fmm-fairness-eval-cli},
  note         = {Zenodo DOI to be minted on the first stable (0.2.0) release}
}

@thesis{pereiro2024dermfairness,
  author = {Pereiro, C{\'e}sar},
  title  = {Foundation-model-based fairness evaluation in dermatology classification using the AI4SkIN dataset},
  school = {Universitat Polit{\`e}cnica de Val{\`e}ncia},
  year   = {2024},
  url    = {https://riunet.upv.es/handle/10251/226903}
}
```

---

## Roadmap

- v0.1 (this release): CLI, 4 fairness gap metrics, composite score, AI Act manifest mode, SHA-256 audit chain.
- v0.2: BCa bootstrap, sub-group intersectionality (`site × sex`), CSV-of-CSVs batch mode.
- v0.3: HTML report option, hosted fairness-CI (Phase 2, gated on validation pass).
- v0.4: subgroup-aware threshold optimisation (opt-in, with the appropriate caveats).

---

## License

MIT. See [LICENSE](LICENSE).
