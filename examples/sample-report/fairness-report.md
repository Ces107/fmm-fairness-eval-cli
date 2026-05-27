# Fairness evaluation report

- **Tool**: fmm-fairness-eval v0.2.0a11
- **Generated (UTC)**: 2026-05-27T16:07:45Z
- **Predictions file**: `examples/predictions.csv`
- **Sample count**: 900
- **Number of classes (K)**: 2
- **Protected attributes**: site, sex, age_bucket
- **Site attribute**: `site`
- **Manifest mode**: `ai-act`

## SaMD composite fairness score
- **Score**: 0.8538  (1.0 = perfectly fair, 0.0 = maximally unfair)
- **Components**: F1_site=0.3605, site_term=0.0649, EO=0.0045, DP=0.0223, CAL=0.0263
- **Effective weights**: {'f1_site': 0.35, 'site': 0.2, 'eo': 0.2, 'dp': 0.1, 'cal': 0.15}

## Inter-site AUC
- **Variance across sites**: 0.001052
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
- **weighted_f1_gap**: 0.3605  [95% CI: 0.3022, 0.4214]
  - `hospital_A` (n=300): 0.9900
  - `hospital_B` (n=300): 0.9433
  - `hospital_C` (n=300): 0.6295
- **macro_f1_gap**: 0.3513  [95% CI: 0.2975, 0.4052]
  - `hospital_A` (n=300): 0.9900
  - `hospital_B` (n=300): 0.9433
  - `hospital_C` (n=300): 0.6386
- **per_class_f1_gap**: 0.4324  [95% CI: 0.3609, 0.5192]  (per-class gaps: c0=0.2703, c1=0.4324)
  - `hospital_A` (n=300): 0.9900
    per-class F1: c0=0.9897, c1=0.9903
  - `hospital_B` (n=300): 0.9433
    per-class F1: c0=0.9439, c1=0.9428
  - `hospital_C` (n=300): 0.6295
    per-class F1: c0=0.7193, c1=0.5579
- **multi_class_auc_gap**: 0.0716  [95% CI: 0.0466, 0.1001]
  - `hospital_A` (n=300): 0.9998
  - `hospital_B` (n=300): 0.9937
  - `hospital_C` (n=300): 0.9282

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
- **weighted_f1_gap**: 0.0055  [95% CI: 0.0000, 0.0202]
  - `F` (n=427): 0.8651
  - `M` (n=473): 0.8596
- **macro_f1_gap**: 0.0053  [95% CI: 0.0000, 0.0188]
  - `F` (n=427): 0.8654
  - `M` (n=473): 0.8601
- **per_class_f1_gap**: 0.0105  [95% CI: 0.0049, 0.0616]  (per-class gaps: c0=0.0105, c1=0.0001)
  - `F` (n=427): 0.8651
    per-class F1: c0=0.8774, c1=0.8535
  - `M` (n=473): 0.8596
    per-class F1: c0=0.8669, c1=0.8533
- **multi_class_auc_gap**: 0.0071  [95% CI: 0.0005, 0.0240]
  - `F` (n=427): 0.9686
  - `M` (n=473): 0.9757

### `age_bucket`
- **equal_opportunity_gap**: 0.0073  [95% CI: 0.0073, 0.1259]
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
- **weighted_f1_gap**: 0.0016  [95% CI: 0.0012, 0.0016]
  - `0-39` (n=272): 0.8623
  - `40-64` (n=360): 0.8628
  - `65+` (n=268): 0.8612
- **macro_f1_gap**: 0.0017  [95% CI: 0.0015, 0.0017]
  - `0-39` (n=272): 0.8624
  - `40-64` (n=360): 0.8634
  - `65+` (n=268): 0.8617
- **per_class_f1_gap**: 0.0097  [95% CI: 0.0097, 0.0888]  (per-class gaps: c0=0.0097, c1=0.0083)
  - `0-39` (n=272): 0.8623
    per-class F1: c0=0.8771, c1=0.8477
  - `40-64` (n=360): 0.8628
    per-class F1: c0=0.8714, c1=0.8555
  - `65+` (n=268): 0.8612
    per-class F1: c0=0.8674, c1=0.8560
- **multi_class_auc_gap**: 0.0176  [95% CI: 0.0061, 0.0405]
  - `0-39` (n=272): 0.9679
  - `40-64` (n=360): 0.9819
  - `65+` (n=268): 0.9643

## Calibration
- **Global Brier score**: 0.2127  (0 = perfectly calibrated; lower is better)
- **Per-class Brier**: c0=0.1064, c1=0.1064
### `site` calibration (Brier gap: 0.3170)
- `hospital_A` (n=300): Brier=0.0799
  - c0: HL p=0.000, c1: HL p=0.000
- `hospital_B` (n=300): Brier=0.1612
  - c0: HL p=0.000, c1: HL p=0.000
- `hospital_C` (n=300): Brier=0.3970
  - c0: HL p=0.000, c1: HL p=0.000
### `sex` calibration (Brier gap: 0.0006)
- `F` (n=427): Brier=0.2130
  - c0: HL p=0.000, c1: HL p=0.000
- `M` (n=473): Brier=0.2124
  - c0: HL p=0.000, c1: HL p=0.000
### `age_bucket` calibration (Brier gap: 0.0126)
- `0-39` (n=272): Brier=0.2132
  - c0: HL p=0.000, c1: HL p=0.000
- `40-64` (n=360): Brier=0.2071
  - c0: HL p=0.000, c1: HL p=0.000
- `65+` (n=268): Brier=0.2197
  - c0: HL p=0.000, c1: HL p=0.000

## Regulatory mapping
- **Framework**: EU AI Act (Regulation 2024/1689)
  - **Art. 9 — Risk management system**
    - Metrics: samd_fairness_score, inter_site_auc_variance, weighted_f1_gap
    - Note: Weighted-F1 inter-site gap is the headline residual-risk indicator; AUC variance gives the K-agnostic generalisation summary.
  - **Art. 10 — Data and data governance**
    - Metrics: equal_opportunity_gap, demographic_parity_gap, calibration_gap, weighted_f1_gap, per_class_f1_gap
    - Note: Per-attribute breakdown evidences Art. 10(2)(f-g) examination of biases and shortcomings. Multi-class deployments use the F1 family; binary deployments retain EO / DP / CAL.
  - **Art. 14 — Human oversight**
    - Metrics: cohen_kappa_matrix, fleiss_kappa, krippendorff_alpha, ai_vs_pooled_raters_kappa
    - Note: Inter-rater agreement statistics (when --rater-cols is provided) document the human-expert reference against which the AI is evaluated, evidencing the Art. 14 human-oversight obligation.
  - **Art. 15 — Accuracy, robustness, cybersecurity**
    - Metrics: inter_site_auc_variance, multi_class_auc_gap, brier_score, hosmer_lemeshow, permutation_p_value, minimum_detectable_effect
    - Note: Site-level AUC variance + OVR-macro AUC gap evidence discrimination generalisation under K-class deployments. Brier score and per-class Hosmer-Lemeshow goodness-of-fit evidence probability-calibration robustness. Permutation p-value + MDE bound the statistical-significance and power of any observed inter-site gap.

## Caveats (read before quoting these numbers)
- Gap metrics depend on the operating threshold (or argmax decision) used to produce y_pred.
- Groups with n < min_group_n were excluded; small-sample bootstrap CIs are approximate.
- Multi-site bias is often confounded with prevalence shift across sites.
- Per-class F1 gaps identify which class drives an inter-site disparity; combine with the per-class breakdown in the per-group rows.
- This tool measures fairness; it does not certify it.

