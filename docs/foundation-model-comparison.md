# Foundation-model comparison mode

The CLI subcommand `fmm-fairness compare` answers the practitioner's
actual decision question: *"which foundation model should I deploy
across my hospital network?"*. The answer is the Pareto frontier over
(overall weighted F1, inter-site weighted F1 gap), plus a single EU
AI Act Art. 9 residual-risk recommendation that picks one frontier
candidate under a configurable fairness floor.

This is the differentiator vs FairLearn / AIF360. Those libraries
provide per-model fairness metrics; neither helps a regulator-facing
operator pick between candidate backbones on a joint
accuracy-and-fairness criterion. The TFG that this CLI is named after
ran six foundation-model embeddings (UNI, CONCH, PLIP, TransPath,
GigaPath, TITAN) plus a CONCH+UNI concatenation; the comparison mode
ships the machinery to make that kind of evaluation reproducible.

---

## Invocation

```
fmm-fairness compare uni.csv conch.csv plip.csv \
    --labels uni,conch,plip \
    --protected-attrs site,sex \
    --site-attribute site \
    --num-classes 6 \
    --fairness-floor 0.10 \
    --output comparison-report/
```

Each input CSV has the same shape as the `evaluate` subcommand input:
`y_true`, `y_pred`, score columns (binary `y_score` or multi-class
`y_score_0..y_score_{K-1}`), plus the declared protected attributes.

The number of input CSVs and the comma-separated `--labels` count must
match.

---

## What the comparison computes

For each candidate, the tool computes:

- `overall_weighted_f1`: support-weighted F1 across all classes (decision
  quality summary).
- `overall_macro_f1`: equal-weight-per-class F1 (rare-class-sensitive
  alternative).
- `weighted_f1_gap_site`: max-min weighted F1 across the site attribute
  (the TFG headline inter-site disparity).
- `macro_f1_gap_site`: same on macro F1.
- `samd_fairness_score`: the composite from `docs/samd-fairness-score.md`.

These five numbers define each candidate's position in the
accuracy-and-fairness plane. Two of them — `overall_weighted_f1` and
`weighted_f1_gap_site` — are the axes of the Pareto frontier.

---

## Pareto frontier semantics

Convention: **higher overall_weighted_f1 is better, lower
weighted_f1_gap_site is better**. A candidate `j` *dominates* candidate
`i` when both:

- `overall_weighted_f1(j) >= overall_weighted_f1(i)`, and
- `weighted_f1_gap_site(j) <= weighted_f1_gap_site(i)`,

with at least one inequality being strict. The Pareto frontier is the
set of candidates that no other candidate dominates.

Dominated candidates should not be deployed as-is. By construction, a
frontier candidate exists that is better on every axis the regulator
cares about. The comparison evidence pack lists the dominated
candidates explicitly so the operator's submission can document the
rejection.

---

## AI Act Art. 9 recommendation

The frontier may contain multiple candidates with different
accuracy-vs-fairness trade-offs. The tool picks one under a
documented heuristic:

1. Filter the frontier to candidates whose
   `weighted_f1_gap_site` is at or below `--fairness-floor`
   (default 0.10).
2. Among the filtered candidates, pick the one with the highest
   `overall_weighted_f1`. This is the recommended candidate.
3. If no frontier candidate meets the floor, fall back to the
   frontier candidate with the smallest gap (the "fairest" frontier
   candidate). The rationale text flags this case explicitly so the
   operator sees the trade-off and can either revise the floor,
   apply site-specific calibration, or pick a different candidate.

EU AI Act Art. 9 (risk management system) asks for documented evidence
that residual risks have been minimised under the declared constraints.
The fairness floor is the operator's declared inter-site disparity
constraint; the frontier is the universe of candidates that the
constraint operates on; the recommended candidate is the frontier
member that meets the constraint with the best accuracy.

---

## Worked example

Three candidates:

| Label   | overall_weighted_f1 | weighted_f1_gap_site |
|---------|---------------------|----------------------|
| `uni`   | 0.75                | 0.05                 |
| `conch` | 0.89                | 0.08                 |
| `plip`  | 0.55                | 0.20                 |

`plip` is dominated by both `uni` (lower accuracy, higher gap) and
`conch` (lower accuracy, higher gap on both axes). It is removed from
the frontier.

`uni` is not dominated by `conch`: `conch` is more accurate but has a
larger gap. `conch` is not dominated by `uni`: `uni` is fairer but less
accurate. Both are on the frontier.

With `--fairness-floor 0.10`, both frontier candidates qualify. The
recommendation picks `conch` (higher overall F1). With
`--fairness-floor 0.06`, only `uni` qualifies and is recommended. With
`--fairness-floor 0.04`, neither qualifies; the fallback rationale
flags this and recommends `uni` (the fairest frontier candidate) for
operator review.

---

## Output

The comparison pack is a directory containing:

- `comparison-report.md`: human-readable Markdown report with the
  frontier list, the recommendation rationale, and the per-model
  summary table.
- `comparison-evidence.json`: machine-readable JSON with the full
  ComparisonResult dict and the tool version.
- `audit.sha256`: SHA-256 of both above, in the same format as the
  `evaluate` subcommand output.

The SHA-256 chain is the tamper-evidence proof for the regulatory
submission. Pin both files plus the audit in the QMS / change-control
record.

---

## Limitations and caveats

1. The tool **does not enforce identical test sets across candidates**.
   The operator's responsibility is to ensure the same items appear in
   each CSV, with the same protected-attribute declarations, so the
   comparison is apples-to-apples. The tool checks parseability and
   class-index ranges but cannot verify identity of the items.
2. **Bootstrap CIs are not propagated into the frontier**. The frontier
   uses point estimates; uncertainty quantification on the frontier
   itself (e.g. "with 95% probability, candidate X is on the
   frontier") is a v0.3 candidate.
3. **The frontier is computed on two axes only**: overall weighted F1
   and inter-site weighted F1 gap. Multi-axis frontiers (also
   incorporating macro F1, calibration, or kappa) are conceptually
   straightforward but would change the recommendation semantics; v0.2
   keeps the two-axis case for interpretability.
4. The default fairness floor of 0.10 is a **starting point**, not a
   regulatory threshold. Choose the floor on clinical-impact grounds:
   if a 5-percentage-point F1 gap between sites is the maximum the
   review committee will sign off on, set `--fairness-floor 0.05`.
5. Plotting is **deferred to S6 of the roadmap** (subgroup
   intersectionality + plotting). The current output is text-only
   (Markdown table + JSON). The data needed to plot a Pareto curve is
   in `comparison-evidence.json` under `result.models`.

---

## References

- The accuracy-vs-fairness Pareto-frontier framing for fair ML is
  standard; see Hardt, Price, Srebro (NeurIPS 2016) for the
  equal-opportunity setting and Kearns et al. (ICML 2018) for the
  rich-subgroups generalisation.
- The clinical motivation for foundation-model comparison comes from
  the pathology literature: Lu et al. 2024 (CONCH, Nat. Med. 30:863),
  Chen et al. 2024 (UNI, Nat. Med. 30:850-862), Wang et al. 2024
  (TransPath, Med. Image Anal. 80), Filiot et al. 2023 (PLIP).
- EU AI Act Art. 9 (Regulation 2024/1689) is the regulatory anchor for
  the residual-risk recommendation.
