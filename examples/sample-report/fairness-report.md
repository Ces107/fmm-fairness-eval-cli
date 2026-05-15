# Fairness evaluation report

- **Tool**: fmm-fairness-eval v0.1.0
- **Generated (UTC)**: 2026-05-14T16:44:59Z
- **Predictions file**: `examples/predictions.csv`
- **Sample count**: 900
- **Protected attributes**: site, sex, age_bucket
- **Site attribute**: `site`
- **Manifest mode**: `ai-act`

## SaMD composite fairness score
- **Score**: 0.9654  (1.0 = perfectly fair, 0.0 = maximally unfair)
- **Components**: site_term=0.0649, EO=0.0045, DP=0.0223, CAL=0.0263

## Inter-site AUC
- **Variance (max-min**)**: 0.001052
  - `hospital_A` (n=300): AUC=0.9998
  - `hospital_B` (n=300): AUC=0.9937
  - `hospital_C` (n=300): AUC=0.9282

## Per-attribute fairness gaps
### `site`
- **equal_opportunity_gap**: 0.6043  [95% CI: 0.5304, 0.6802]
  - `hospital_A` (n=300): 0.9935
  - `hospital_B` (n=300): 0.9091
  - `hospital_C` (n=300): 0.3892
- **demographic_parity_gap**: 0.2967  [95% CI: 0.2300, 0.3701]
  - `hospital_A` (n=300): 0.5167
  - `hospital_B` (n=300): 0.4767
  - `hospital_C` (n=300): 0.2200
- **calibration_gap**: 0.0875  [95% CI: 0.0504, 0.1282]
  - `hospital_A` (n=300): 0.1603
  - `hospital_B` (n=300): 0.2069
  - `hospital_C` (n=300): 0.2478

### `sex`
- **equal_opportunity_gap**: 0.0016  [95% CI: 0.0015, 0.0895]
  - `F` (n=427): 0.7545
  - `M` (n=473): 0.7529
- **demographic_parity_gap**: 0.0165  [95% CI: 0.0012, 0.0812]
  - `F` (n=427): 0.3958
  - `M` (n=473): 0.4123
- **calibration_gap**: 0.0221  [95% CI: 0.0008, 0.0478]
  - `F` (n=427): 0.1542
  - `M` (n=473): 0.1763

### `age_bucket`
- **equal_opportunity_gap**: 0.0073  [95% CI: 0.0124, 0.1259]
  - `0-39` (n=272): 0.7518
  - `40-64` (n=360): 0.7513
  - `65+` (n=268): 0.7586
- **demographic_parity_gap**: 0.0282  [95% CI: 0.0095, 0.1156]
  - `0-39` (n=272): 0.3897
  - `40-64` (n=360): 0.4056
  - `65+` (n=268): 0.4179
- **calibration_gap**: 0.0305  [95% CI: 0.0085, 0.0645]
  - `0-39` (n=272): 0.1503
  - `40-64` (n=360): 0.1808
  - `65+` (n=268): 0.1666

## Regulatory mapping
- **Framework**: EU AI Act (Regulation 2024/1689)
  - **Art. 9 — Risk management system**
    - Metrics: samd_fairness_score, inter_site_auc_variance
    - Note: Inter-site performance variance is a residual risk requiring documented mitigation.
  - **Art. 10 — Data and data governance**
    - Metrics: equal_opportunity_gap, demographic_parity_gap, calibration_gap
    - Note: Per-attribute breakdown evidences Art. 10(2)(f-g) examination of biases and shortcomings.
  - **Art. 15 — Accuracy, robustness, cybersecurity**
    - Metrics: inter_site_auc_variance
    - Note: Site-level AUC variance evidences performance generalization claims.

## Caveats (read before quoting these numbers)
- Equal-opportunity gap depends on the operating threshold used to produce `y_pred`.
- Groups with n < min_group_n were excluded; small-sample bootstrap CIs are approximate.
- Multi-site bias is often confounded with prevalence shift across sites.
- This tool measures fairness; it does not certify it.

