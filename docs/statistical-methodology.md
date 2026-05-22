# Statistical methodology

v0.2 turns the v0.1 percentile-bootstrap CI into a defensible research-
grade inference layer. The roadmap S4 deliverables landed in the
`fmm_fairness.statistics` module and are wired into the F1-family
metrics by default:

- Bias-corrected and accelerated (**BCa**) bootstrap CI (Efron 1987).
- Label-shuffle **permutation test** for `H0 = no gap`.
- **Minimum detectable effect (MDE)** at alpha=0.05, power=0.80.
- **Cohen's d** and **odds ratio** effect-size summaries (helpers; not
  emitted in the per-attribute block by default but available through
  the Python API).

The headline shift: every gap reported by `weighted_f1_gap` and
`macro_f1_gap` now ships a BCa CI, a (optional) permutation p-value,
and an MDE statement. The dossier-grade question regulators actually
ask ("is the disparity statistically credible given the sample?") is
answerable from the JSON evidence pack alone.

---

## 1. BCa bootstrap

The percentile bootstrap (v0.1) returns the alpha/2 and 1 - alpha/2
quantiles of the bootstrap distribution. It is unbiased only when the
sampling distribution of the estimator is symmetric and centred on the
true value. For F1 gaps under inter-site disparity, neither assumption
holds.

The bias-corrected and accelerated (BCa) interval applies two
adjustments to the percentile endpoints:

- A **bias correction** `z0` derived from the proportion of bootstrap
  replicates below the original estimate. When the bootstrap
  distribution is symmetric, `z0 = 0` and the BCa endpoints collapse
  to the percentile endpoints.
- An **acceleration** `a` derived from the jackknife (leave-one-out)
  influence function. When the influence function is constant,
  `a = 0`.

The adjusted endpoints are:

```
alpha1 = Phi(z0 + (z0 + z_{alpha/2})    / (1 - a * (z0 + z_{alpha/2})))
alpha2 = Phi(z0 + (z0 + z_{1-alpha/2})  / (1 - a * (z0 + z_{1-alpha/2})))
```

where `Phi` is the standard normal CDF and `z_p` is the standard
normal quantile.

The implementation falls back to the percentile interval when the BCa
endpoints are degenerate (zero-variance jackknife, infinite bias
correction). This keeps the tool robust on small samples while still
emitting the methodologically-correct interval when the sample size
allows.

### When to override to percentile

Set `--bootstrap-method percentile` when:

1. Reproducing a v0.1 number for direct comparison.
2. The sample is so small that the jackknife is unstable
   (`bootstrap_se` will flag this in the JSON).
3. The downstream reviewer specifically asked for the simpler
   percentile interval (e.g. a journal style guide).

BCa is the default because for any production-grade SaMD validation,
the bias and skewness of the gap distribution are real and the
percentile interval misstates the location of the CI.

---

## 2. Permutation test

The null hypothesis `H0 = no gap` is tested by shuffling the protected-
attribute labels uniformly at random across the items, recomputing the
max-min gap on each permutation, and computing the proportion of
permutations whose statistic is at least as extreme as the observed.

The p-value is reported with the standard `(n_extreme + 1) / (n_iters + 1)`
plug-in to avoid `p = 0`. Default `--permutation-iters 0` keeps the
test off (it doubles the runtime of the F1-family metrics) — opt in
with `--permutation-iters 1000` when the regulatory dossier needs a
formal hypothesis test.

The test is two-sided in spirit (gap is non-negative by construction,
so "extreme" means "as large as or larger than observed"); the
interpretation is "the observed gap would have appeared by chance with
probability p".

### Caveats

- The permutation test conditions on the **observed marginal label
  distribution**. It is not a test of "would another sample show the
  same gap"; it is a test of "could this sample's gap come from a
  no-gap data-generating process".
- Multi-attribute testing requires a multiple-comparisons correction
  (Bonferroni or Benjamini-Hochberg) when the dossier reports p-values
  for many attributes. The tool does not apply this correction
  automatically; the operator must report it in the submission.

---

## 3. Minimum detectable effect (MDE)

The MDE answers the operator's pre-experiment design question: "given
the current sample sizes, how big a gap would I be able to detect at
alpha=0.05, power=0.80?". The formula:

```
MDE = (z_{1 - alpha/2} + z_{power}) * SE_gap
```

where `SE_gap` is the bootstrap-estimated standard error of the max-
min gap statistic.

This is the standard sample-size formula applied to the bootstrap SE
rather than to a closed-form SE. It is the most defensible "post-hoc-
adjacent" power statement available without re-doing the experiment.
Three caveats:

- Post-hoc power is a controversial topic in statistics. We report
  the MDE rather than "observed power" precisely because the MDE is a
  characterisation of the *sample*, not of the *result*.
- The bootstrap SE is itself estimated; for tiny samples the MDE
  carries propagated uncertainty we do not currently expose.
- The MDE assumes a normal sampling distribution for the gap, which is
  approximately true under the central limit theorem at the n's the
  tool typically sees, and is double-checked by the BCa adjustment.

The output JSON carries `minimum_detectable_effect`, `alpha`, and
`power` alongside the CI for each F1-family gap.

---

## 4. Cohen's d and odds ratio (helpers)

The `cohens_d(values_a, values_b)` function returns the standardised
mean difference between two groups using pooled SD. Use it when
comparing per-rater scores, per-site continuous outcomes, or any other
two-group continuous comparison.

The `odds_ratio_binary(success_a, total_a, success_b, total_b)`
function returns the odds ratio for a 2x2 success/failure
contingency table. Use it for binary outcome comparisons (e.g. AI
correct vs human correct across two sites).

Both are helpers for the Python API. They are not emitted into the
per-attribute evidence block by default because their semantics are
specific to the comparison the operator is making (between which two
groups? on which axis?). The operator is expected to call them in a
notebook for the specific comparison the dossier requires.

---

## 5. CLI knobs

The full inference layer is controlled by five new flags on `fmm-fairness evaluate`:

| Flag                  | Default     | Meaning                                                |
|-----------------------|-------------|--------------------------------------------------------|
| `--bootstrap-method`  | `bca`       | CI method: `bca` (default) or `percentile`             |
| `--bootstrap-iters`   | `1000`      | Bootstrap iteration count                              |
| `--permutation-iters` | `0`         | Permutation iteration count; 0 disables the test        |
| `--alpha`             | `0.05`      | Significance level for the CI and MDE                  |
| `--power`             | `0.80`      | Target power for the MDE                               |

Example invocation reproducing the TFG-style full inference pack:

```
fmm-fairness evaluate predictions.csv \
    --protected-attrs site,sex \
    --site-attribute site \
    --num-classes 6 \
    --bootstrap-method bca \
    --bootstrap-iters 2000 \
    --permutation-iters 2000 \
    --alpha 0.05 \
    --power 0.80 \
    --output ai4skin-report/
```

---

## 6. AI Act mapping

The new inference layer strengthens two AI Act articles:

- **Art. 15 (accuracy, robustness, cybersecurity)**: the BCa CI is the
  proper uncertainty quantification for the fairness claims under
  Art. 15. A submission that reports a point estimate without a CI
  invites the reviewer to substitute their own (worse) bound.
- **Art. 9 (risk management system)**: the permutation p-value and the
  MDE collectively answer the "did we power the study sufficiently to
  detect the residual risk we are committing to?" question that
  Art. 9 demands.

---

## 7. References

- Efron, B. (1987). *Better Bootstrap Confidence Intervals.*
  J. Am. Stat. Assoc. 82(397):171-185.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap.*
  Chapman & Hall.
- DiCiccio, T. J., Efron, B. (1996). *Bootstrap Confidence Intervals.*
  Statistical Science 11(3):189-228.
- Ernst, M. D. (2004). *Permutation Methods: A Basis for Exact Inference.*
  Statistical Science 19(4):676-685.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
  Sciences.* 2nd ed., Lawrence Erlbaum.
- Hoenig, J. M., Heisey, D. M. (2001). *The Abuse of Power: The
  Pervasive Fallacy of Power Calculations for Data Analysis.* The
  American Statistician 55(1):19-24. (The "observed power" critique;
  motivates the MDE-not-observed-power choice.)
