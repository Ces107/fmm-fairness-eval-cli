# Intersectional fairness and calibration depth (S6)

This document covers two complementary upgrades that land together in
`fmm-fairness-eval` v0.2.0a6: intersectional cross-product fairness and the
calibration block (Brier score, Hosmer-Lemeshow goodness-of-fit, reliability
diagrams).

## 1. Why these two together

Single-axis fairness misses intersectional disparities. A model can pass an
inter-site gap test and an inter-sex gap test independently and still produce
a meaningfully worse decision rate for women at one specific site. Likewise,
discrimination (F1, AUC) tells you the model's ordering of cases is right;
calibration tells you the probabilistic claim attached to each case is right.
A model that is discriminative but poorly calibrated produces risk scores that
downstream decision rules cannot trust as probabilities; the EU AI Act
Art. 15 robustness requirement is most often read as discrimination plus
calibration jointly.

## 2. Intersectional fairness

### 2.1 CLI

```text
fmm-fairness evaluate predictions.csv \
    --protected-attrs site,sex,age_bucket \
    --intersect "site*sex,site*age_bucket" \
    --output report/
```

The `--intersect` flag takes a comma-separated list of cross-products. Each
cross-product is written as `axis1*axis2[*axis3]` using axis names that have
already been declared via `--protected-attrs`. Internally each cross-product
becomes a synthetic attribute, joined by `*`, e.g. `A*F`, `A*M`, `B*F`, `B*M`.
The synthetic attribute is then run through the same weighted-F1-gap and
macro-F1-gap pipeline used for single-axis attributes.

The evidence pack ships a new top-level block:

```text
"intersectional_breakdown": {
    "intersections_declared": [["site","sex"], ["site","age_bucket"]],
    "min_group_n": 20,
    "shrinkage_kappa": 0,
    "shrinkage_pivot": 50,
    "results": [
        {
            "axes": ["site","sex"],
            "synthetic_attribute": "_intersect_site*sex",
            "weighted_f1_gap": { "gap": 0.4173, "per_cell": [...], ...},
            "macro_f1_gap":    { "gap": 0.3987, "per_cell": [...], ...}
        }, ...
    ]
}
```

### 2.2 Small-cell handling

Cross-product cells are smaller than their single-axis parents by
construction. The same `min_group_n` guard used elsewhere (default 20) prunes
cells below the threshold and surfaces them under `excluded_cells`. A warning
is also raised so operators do not silently lose data.

For cells in the middle band (default `20 <= n < 50`) the tool supports
optional Bayesian shrinkage. With `--shrinkage-kappa K > 0`, sub-pivot cell
values are pulled toward the global F1 over retained cells using the
empirical-Bayes weighted mean:

```text
v_shrunk = (n * v_empirical + kappa * v_global) / (n + kappa)
```

The shrunk cells are listed in `shrunk_cells` for audit transparency. Default
shrinkage is `kappa = 0`: numbers stay empirical unless an operator opts in.

### 2.3 What the metric tells you

A cross-product gap larger than its single-axis components is the signal that
the disparity is genuinely intersectional. The gap is the max-min over
cells, so a single underperforming cell can dominate the headline. Always
read the gap together with the `per_cell` list; a 0.40 gap driven by a
single n=21 cell is a different story than a 0.40 gap with three large cells
clearly underneath the rest.

## 3. Calibration block

The calibration block is always emitted, regardless of whether
`--intersect` was used. It carries:

- `global_brier_score` and `global_per_class_brier`: dataset-wide reference.
- `per_attribute`: per-group Brier + per-class Brier vector, plus per-class
  Hosmer-Lemeshow and per-class reliability-bin data, per protected attribute.

### 3.1 Multi-class Brier score

Following Brier 1950 and Murphy 1973, the multi-class Brier score is

```text
mean over samples i of  sum_k (y_onehot_{i,k} - p_{i,k})^2
```

For K=2 this matches the standard binary Brier `mean((y - p)^2)` up to a
factor of two; the implementation always uses the multi-class form so K=2
deployments and K>=3 deployments share one number. The score is in `[0, 2]`;
zero means perfect calibration on perfect ordering, two means probability one
on the wrong class everywhere.

### 3.2 Hosmer-Lemeshow goodness of fit

The Hosmer-Lemeshow C-statistic is the canonical goodness-of-fit test for
binary calibration. With g equal-frequency bins ordered by predicted
probability, the statistic is

```text
chi2 = sum_{g} (O_g - E_g)^2 / [ E_g * (1 - E_g / n_g) ]
```

distributed approximately chi-square with `g - 2` degrees of freedom. The
implementation reports `chi2`, `df`, and `p_value` per class (one-vs-rest for
K >= 3). Caveats:

- The test is sensitive to bin construction; equal-frequency deciles are
  the standard choice and are used here.
- Bins with `n < 2` are merged into their predecessor to keep the chi-square
  sum defined.
- The test is undefined when y_true is constant or all bins are degenerate;
  the implementation returns NaN with a `note` field that records the cause.
- A non-significant p-value is not evidence of calibration; with small N
  the test is underpowered. Read it alongside the Brier score and reliability
  bins.

### 3.3 Reliability bins (and optional plots)

Each group gets a per-class array of `n_reliability_bins` equal-width bins
over `[0, 1]` with `count`, mean predicted probability (`confidence`), and
observed positive rate (`accuracy`). These bins are always in the JSON
evidence; they are everything a downstream consumer needs to render a
reliability diagram themselves.

When `--render-plots` is passed and `matplotlib` is installed (via the
`fmm-fairness-eval[plots]` optional dependency), the CLI writes one PNG per
`(attribute, group, class)` to `<output>/plots/` and records the paths under
`rendered_plots` in the evidence pack. Without matplotlib the flag is a
no-op; a warning is issued and the JSON / Markdown / audit hash still emit.

## 4. AI Act Art. 15 mapping

The calibration block extends the `--manifest-mode ai-act` mapping for
Art. 15 (Accuracy, robustness, cybersecurity) with three additional cited
metrics: `brier_score`, `hosmer_lemeshow`, and the Art. 15 reading of the
permutation p-value plus MDE from S4. The rationale is that Art. 15 is the
joint discrimination + calibration claim about model behaviour at the
declared deployment population; documenting the calibration statistic
alongside the discrimination gap closes the most common Art. 15 audit
finding (gap reported, calibration silent).

## 5. Default behaviour

Defaults are conservative and additive:

- `--intersect` is off by default; behavior of pre-S6 callers is unchanged.
- `--min-group-n` defaults to 20 (was 20 internally; now also surfaced to
  the CLI).
- `--shrinkage-kappa` defaults to 0 (no shrinkage); shrunk numbers do not
  silently appear in reports.
- The calibration block is always emitted (it is purely additive in JSON +
  Markdown and is cheap to compute). Pre-S6 readers ignore unknown keys.
- `--render-plots` requires the `[plots]` optional dependency. Numeric data
  is unaffected by this flag.

## 6. References

- Brier, G.W. 1950. *Verification of forecasts expressed in terms of
  probability.* Monthly Weather Review 78:1-3.
- Murphy, A.H. 1973. *A new vector partition of the probability score.*
  Journal of Applied Meteorology 12:595-600.
- Hosmer, D.W. and Lemeshow, S. 2000. *Applied Logistic Regression.*
  2nd ed., Wiley. (HL goodness-of-fit test.)
- Niculescu-Mizil, A. and Caruana, R. 2005. *Predicting good probabilities
  with supervised learning.* ICML.
- Crenshaw, K. 1989. *Demarginalizing the intersection of race and sex.*
  University of Chicago Legal Forum. (Origin of intersectionality.)
- Efron, B. and Morris, C. 1973. *Stein's estimation rule and its
  competitors.* JASA 68(341). (Empirical-Bayes shrinkage motivation.)
