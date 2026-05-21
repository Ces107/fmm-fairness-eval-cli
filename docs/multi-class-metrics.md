# Multi-class fairness metrics

This document defines the four multi-class-aware fairness metrics shipped in
`fmm_fairness.metrics` from v0.2 onwards. The motivation is concrete: the
underlying TFG ran a six-class dermatopathology classifier across two centres
(HCUV and HUSC) and reported a weighted-F1 inter-site gap of 0.19. The v0.1
CLI could not reproduce that number because it only spoke binary. v0.2 closes
the gap.

The CLI auto-detects the number of classes K from the score columns
(`y_score_0..y_score_{K-1}`) or, in the binary case, from the single
`y_score` column. The auto-detected value is cross-checked against
`max(y_true) + 1`; an explicit `--num-classes` flag overrides with a
warning on mismatch.

---

## Input contract

Binary deployments keep the v0.1 shape:

| column        | type           | meaning                |
|---------------|----------------|------------------------|
| `y_true`      | int in {0, 1}  | ground-truth label     |
| `y_pred`      | int in {0, 1}  | thresholded prediction |
| `y_score`     | float in [0, 1] | P(class = 1)          |

Multi-class deployments use one column per class for probabilities:

| column          | type              | meaning              |
|-----------------|-------------------|----------------------|
| `y_true`        | int in {0..K-1}   | ground-truth label   |
| `y_pred`        | int in {0..K-1}   | argmax prediction    |
| `y_score_0`     | float in [0, 1]   | P(class = 0)         |
| `y_score_1`     | float in [0, 1]   | P(class = 1)         |
| ...             | ...               | ...                  |
| `y_score_{K-1}` | float in [0, 1]   | P(class = K-1)       |

Rows of probabilities should sum to approximately 1.0 (the tool does not
hard-renormalise; calibration-aware metrics rely on the original numbers).

A binary CSV using the K=2 multi-class shape (`y_score_0`, `y_score_1`) is
also valid; the binary single-column shape stays the canonical form for
backward compatibility with v0.1 fixtures.

---

## `weighted_f1_gap(df, attribute)`

The support-weighted F1 score is computed per group, then the across-group
max-minus-min is reported.

For each group `g`:

```
weighted_F1(g) = sum_{k in 0..K-1} (support(k|g) / N(g)) * F1_k(g)
```

where `F1_k(g)` is the per-class F1 score for class `k` restricted to group
`g`. The headline gap is:

```
weighted_F1_gap(attribute) = max_g weighted_F1(g) - min_g weighted_F1(g)
```

Why this metric: support-weighted F1 is the standard "real-world deployment"
summary on imbalanced clinical cohorts, because it does not over-weight rare
disease classes. Inter-site disparity in this metric is the question a
deployment review committee actually asks: *"if we install this at HUSC,
will it perform like it did at HCUV?"* The v0.1 binary CLI could not answer
that question without throwing away the multi-class structure of the
problem.

Bootstrap CI: percentile bootstrap over a stratified per-group resample;
BCa lands in S4 of the roadmap.

---

## `macro_f1_gap(df, attribute)`

The macro F1 score is computed per group (equal weight per class), then the
across-group max-minus-min is reported. Same shape as `weighted_f1_gap`.

Why a separate metric: macro F1 is the rare-class-sensitive complement to
support-weighted F1. A site that performs well on the three common
dermatopathology classes and poorly on the three rare classes scores high
on weighted F1 (because the rare classes carry little weight) and low on
macro F1. Reporting both surfaces the asymmetry that a single number would
hide.

In medical-AI evaluation the choice between weighted and macro F1 is rarely
a tooling question; it is a clinical-impact question. The minority class
might be the one a missed diagnosis kills.

---

## `per_class_f1_gap(df, attribute)`

The richest of the four. Returns:

- `per_group[i].per_class`: a length-K vector of per-class F1 scores for
  the `i`-th group.
- `per_class_gap`: a length-K vector of across-group max-min gaps, one per
  class.
- `gap`: the worst entry of `per_class_gap` — i.e. *"which single class
  drives the inter-site disparity, and how big is the disparity for that
  class?"*.
- `gap_ci_low/high`: a percentile-bootstrap CI on the worst-class gap.

Why this metric: the headline "weighted F1 gap = 0.19" hides causal
structure. The TFG result was driven by a small number of rare-class
collapses; aggregate metrics would never have surfaced that. Regulators
asking the Art. 10(2)(f-g) question — *which biases or shortcomings were
identified?* — want the per-class breakdown, not a single scalar.

The chosen scalar (`gap = max(per_class_gap)`) is the minimum-detail
summary that does not lie: it is high if and only if at least one class
exhibits a large disparity. Combine with the per-class breakdown for the
full picture.

---

## `multi_class_auc_gap(df, attribute)`

For K=2 this is the binary AUC across groups (max minus min).

For K>2 it is the one-vs-rest macro AUC across groups: for each group `g`,
compute `roc_auc_score(y_true, y_score_matrix, multi_class="ovr",
average="macro")`, then take the across-group max-minus-min.

Why this exists alongside `inter_site_auc_variance`: the variance metric is
the single scalar a QMS dashboard wants; the gap is the per-group
breakdown a fairness review reads. The same K-aware AUC computation
underlies both; they just project to different summaries.

Bootstrap CI: percentile, same mechanism as the other gap metrics.

---

## `inter_site_auc_variance` (now K-aware)

K=2 path: identical numerics to v0.1. The binary `y_score` column is read
directly and the AUC is computed by `roc_auc_score(y_true, y_score)`. No
backward-compatibility break.

K>2 path: the per-site AUC is the OVR macro AUC over the K probability
columns. The reported `gap` field carries the variance (range
≈ `[0, 0.25]`), not the max-min — same semantics as v0.1.

---

## How the binary-only metrics behave under K>2

`equal_opportunity_gap`, `demographic_parity_gap`, and `calibration_gap`
are defined for binary outcomes. Their multi-class extensions (per-class
TPR, per-class selection-rate, per-class calibration) require an explicit
per-class operating threshold which the CLI does not yet ingest in v0.2.
Calling them on K>2 data raises a clear `ValueError` with the recommended
replacement (`weighted_f1_gap`, `macro_f1_gap`, or `per_class_f1_gap`).

The evidence pack auto-routes: binary CSVs receive the EO / DP / CAL
block in `per_attribute_metrics`; multi-class CSVs receive only the
F1-family block.

This split is deliberate. A future release (S6 in the roadmap) introduces
per-class operating-threshold ingestion and revisits whether to extend
EO / DP / CAL into the multi-class case. The choice was *not* to silently
binarise, because silently binarising hides a regulator-facing modelling
choice.

---

## Composite score update

`samd_fairness_score` now takes a fifth component, `F1_SITE`, the
`weighted_f1_gap` at the site attribute. The default weights are:

```
w_f1_site = 0.35
w_site    = 0.20
w_eo      = 0.20   (binary only)
w_dp      = 0.10   (binary only)
w_cal     = 0.15   (binary only)
```

When the input is multi-class, the binary-only weights collapse to zero
and `{w_f1_site, w_site}` are re-normalised to sum to 1 over their pair.
This preserves the `[0, 1]` range of the composite without distorting
the *relative* weighting of the two terms that are defined for any K.
The full sensitivity argument and worked examples live in
`docs/samd-fairness-score.md`.

---

## Reproducing the TFG headline number

The roadmap's S5 ships a fully redacted AI4SkIN-shaped prediction set
under `examples/ai4skin-replication/`. With v0.2 plus that example, the
CLI command:

```
fmm-fairness evaluate examples/ai4skin-replication/predictions.csv \
    --protected-attrs site,sex \
    --site-attribute site \
    --num-classes 6 \
    --output ai4skin-report/
```

emits a `weighted_f1_gap` within 0.005 of the published 0.19 figure.
Until S5 lands, the synthetic 6-class fixture in
`tests/test_multi_class.py::_make_ai4skin_shaped_df` exercises the same
code path.
