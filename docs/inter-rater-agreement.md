# Inter-rater agreement (clinician vs AI)

Inter-rater agreement is the SaMD gold standard for human-oversight
evidence. The underlying TFG evaluated a six-class dermatopathology
classifier against ten pathologists' annotations and reported the
Cohen kappa matrix, the AI-vs-pooled-raters kappa, and per-site
breakdowns. v0.2 ships the same machinery: pairwise Cohen kappa,
Fleiss kappa, Krippendorff alpha (nominal), and a bootstrap CI on the
AI-vs-pooled scalar.

This document defines each statistic, explains why each is shipped, and
gives the Landis-Koch interpretation cut-offs for the kappa-family
outputs.

---

## Input shape

Rater columns are integer columns in the predictions CSV. Each row is one
item (e.g. one whole-slide image, one X-ray); each rater column carries
that rater's class assignment in `{0..K-1}`. The reserved sentinel value
(default `-1`, matching the TFG convention) marks an unrated cell.

CLI invocation:

```
fmm-fairness evaluate predictions.csv \
    --protected-attrs site \
    --site-attribute site \
    --num-classes 6 \
    --rater-cols doc1,doc2,doc3,doc4,doc5,doc6,doc7,doc8,doc9,doc10 \
    --rater-missing-value -1 \
    --output ai4skin-report/
```

The evidence pack gains an `inter_rater_agreement` block with all four
statistics, plus a per-site stratified Cohen kappa matrix when the site
attribute is present.

---

## 1. Cohen kappa pairwise matrix

Cohen kappa measures agreement between two raters, corrected for the
agreement expected by chance. Range: `[-1, 1]` (negative means worse than
chance, 0 means chance-level, 1 means perfect).

The CLI computes one Cohen kappa per rater pair (and one per
rater-vs-AI pair when an `ai_col` is provided), restricted to rows where
both raters supplied a rating (the sentinel value is dropped pairwise).
Diagonal entries are 1.0 by construction.

This matrix is the *direct* evidence regulators ask for under Art. 14
(human oversight) of the EU AI Act: it shows the human-expert
disagreement structure that the AI is being evaluated *against*. A model
whose AI-vs-rater kappa is comparable to rater-vs-rater kappa is, in a
defensible sense, "as good as a typical clinician" on this task. A model
whose kappa is materially lower is not.

### Pairwise vs pooled

The matrix is *pairwise*: one scalar per rater pair, robust to a single
rater having idiosyncratic disagreement. Aggregating to a single
scalar via averaging is misleading because pairwise kappas are not
independent — see Fleiss kappa and Krippendorff alpha below for the
correct global summaries.

---

## 2. Fleiss kappa

Fleiss kappa generalises Cohen kappa to a fixed-N panel of raters and
gives a single agreement scalar. It assumes a constant number of raters
per item, so items where any rater is missing are excluded. Range:
`[-1, 1]` with the same chance-corrected semantics.

Use Fleiss kappa when:

- You have a complete rater panel (all raters rated every item).
- You want one scalar summarising "do the raters basically agree?"

Use Krippendorff alpha when:

- Some raters skipped items (the TFG case: not every pathologist rated
  every slide).
- You want a measure that handles partially-missing ratings without
  dropping the affected items.

---

## 3. Krippendorff alpha (nominal scale)

Krippendorff alpha is the missing-tolerant generalisation. v0.2 implements
the nominal-scale version: the difference function is
`delta(a, b) = 1 if a != b else 0`. Observed disagreement is computed per
item over the actually-rated pairs; expected disagreement is computed from
the marginal frequencies of valid ratings. Range: `[-∞, 1]` in principle,
though negative values below `-1` are pathological in practice.

The implementation follows Krippendorff (2011):

```
alpha = 1 - (observed_disagreement / expected_disagreement)
```

Items with fewer than two valid ratings contribute nothing. The marginal
distribution is computed over all valid ratings across all items.

Ordinal and interval-level versions of alpha (which weigh near-misses
less than far-misses) are out of scope for v0.2; multi-class
dermatopathology categories are nominal, so the nominal version is the
correct default.

---

## 4. AI vs pooled-raters Cohen kappa

The headline scalar for SaMD validation. Procedure:

1. For each item, compute the pooled rater label as the **majority vote**
   over non-missing rater columns; ties break to the lowest class index.
   Items where every rater is missing contribute nothing.
2. Compute Cohen kappa between the AI prediction column and the pooled
   label vector.
3. Bootstrap the point estimate with `bootstrap_iters` percentile-bootstrap
   draws (resampling at the item level); report the 2.5th and 97.5th
   percentiles as the 95% CI.

The headline interpretation: *"the AI agrees with the consensus of the
clinicians at kappa = X (95% CI [low, high])."*

### Why pooled rather than per-rater

The per-rater AI kappas are already in the Cohen kappa matrix. The pooled
scalar is a different question: it asks how the AI compares to the
collective clinical judgement, not to any individual rater. Both are
valuable; the matrix supplies pairwise structure, the pooled scalar
supplies a defensible headline number for a QMS dashboard.

### Tie-break choice

Tying votes break to the lowest class index. This is deterministic and
documented, and matches the TFG convention. Alternative tie-breakers
(random, exclude tied items, weighted by rater seniority) are not
shipped because they introduce non-reproducibility or scope creep. If
the operator's clinical context demands a different rule, override at
the call-site via the `pool_raters` helper.

---

## 5. Per-site stratification

When a site attribute is present in the CSV and the CLI is run with
`--site-attribute site`, the evidence pack additionally carries a
per-site Cohen kappa matrix under
`inter_rater_agreement.cohen_kappa_matrix_by_stratum`. This surfaces
inter-site differences in rater agreement that a single global matrix
would average out — e.g., raters from HCUV may agree more with each
other than with raters from HUSC.

The per-site matrix is the same structure as the global matrix, one
copy per site. The Markdown report references it but only renders the
global matrix inline to keep the report regulator-readable; the full
per-stratum data lives in the JSON evidence file.

---

## 6. Landis-Koch interpretation

Landis & Koch (1977) is the canonical interpretation guide for kappa-
family statistics:

| Range          | Strength of agreement |
|----------------|-----------------------|
| < 0.00         | Poor (worse than chance) |
| 0.00 - 0.20    | Slight                |
| 0.21 - 0.40    | Fair                  |
| 0.41 - 0.60    | Moderate              |
| 0.61 - 0.80    | Substantial           |
| 0.81 - 1.00    | Almost perfect        |

Two caveats worth surfacing in a regulator's report:

1. Kappa is sensitive to the class prevalence distribution. Two studies
   with the same underlying agreement structure can produce different
   kappa values if their class prevalences differ ("the kappa paradox").
2. Landis-Koch cut-offs are conventions, not statistical thresholds. A
   defensible SaMD validation argument cites the cut-offs to *contextualise*
   the kappa value, not to *certify* the model.

The evidence pack ships the raw values and a CI. The interpretation is
the operator's regulatory submission to draft, not the tool's to assert.

---

## 7. Regulatory mapping

Art. 14 (human oversight) of the EU AI Act (Regulation 2024/1689) is the
direct anchor: inter-rater agreement statistics document the human-expert
reference against which the AI is evaluated. The evidence pack, when
`--manifest-mode ai-act` is set, cross-cites:

- `cohen_kappa_matrix` -> Art. 14 (pairwise human-disagreement structure)
- `fleiss_kappa`, `krippendorff_alpha` -> Art. 14 (global panel agreement)
- `ai_vs_pooled_raters_kappa` -> Art. 14 (AI calibration against the
  human consensus, the headline oversight metric)

Art. 15 (accuracy, robustness) is the secondary mapping when the AI-vs-
pooled kappa is reported with its bootstrap CI: the CI quantifies the
robustness of the headline claim under resampling.

---

## 8. References

- Cohen, J. (1960). *A coefficient of agreement for nominal scales.*
  Educational and Psychological Measurement 20(1):37-46.
- Fleiss, J. L. (1971). *Measuring nominal scale agreement among many
  raters.* Psychological Bulletin 76(5):378-382.
- Krippendorff, K. (2011). *Computing Krippendorff's alpha reliability.*
  Annenberg School for Communication, University of Pennsylvania.
  https://repository.upenn.edu/asc_papers/43
- Landis, J. R., Koch, G. G. (1977). *The measurement of observer
  agreement for categorical data.* Biometrics 33(1):159-174.
- Feinstein, A. R., Cicchetti, D. V. (1990). *High agreement but low
  kappa: I. The problems of two paradoxes.* Journal of Clinical
  Epidemiology 43(6):543-549. (The "kappa paradox" reference.)
