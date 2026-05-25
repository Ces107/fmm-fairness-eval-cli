# EU AI Act dossier mapping (S7)

`fmm-fairness-eval` v0.2.0a7 introduces `--manifest-mode ai-act-full`, an
extension of the basic `ai-act` regulatory mapping that adds three more
articles (Art. 13 transparency, Art. 72 post-market monitoring, and a
consolidated Art. 14 cross-cite) plus a bundled template pack at
`src/fmm_fairness/templates/ai_act/` that the dossier references.

## 1. Why two manifest modes

The old `--manifest-mode ai-act` is preserved unchanged: it surfaces a
short cross-reference from the metric blocks to four AI Act articles
(Art. 9, 10, 14, 15). That mode is the right choice for a quick technical
note. The new `--manifest-mode ai-act-full` is the right choice when the
provider is writing a conformity-assessment dossier and needs the
documentary shell around the numbers.

`ai-act-full` is strict superset of `ai-act`: the basic `regulatory_mapping`
block is still emitted; the new `ai_act_full` block is added alongside it,
with cross-references to the bundled templates.

## 2. CLI

```text
fmm-fairness evaluate predictions.csv \
    --protected-attrs site,sex \
    --rater-cols doc1,doc2,...,doc10 \
    --manifest-mode ai-act-full \
    --model-card path/to/model-card.yaml \
    --intersect "site*sex" \
    --output report/
```

The `--model-card` flag is optional; when present, the parsed YAML / JSON
is embedded under `ai_act_full.articles[Art. 13].model_card`. When absent,
Art. 13 is marked `model_card_present: false` so an auditor sees the gap
explicitly rather than implicitly.

`--rater-cols` is independent but recommended: with no rater columns the
Art. 14 block is documentary only. With rater columns, the inter-rater
agreement statistics from S2 (`ai_vs_pooled_raters_kappa` + the kappa
matrix) become the numeric evidence under Art. 14.

## 3. The six articles covered

### Art. 9 — Risk management system

Mapped metrics: `samd_fairness_score`, `inter_site_auc_variance`,
`weighted_f1_gap`, `intersectional_weighted_f1_gap`.

The composite SaMD fairness score is the operator-facing single number;
the inter-site weighted F1 gap (with its BCa CI from S4) is the headline
residual-risk indicator. Intersectional cross-products surface compounding
disparities that the single-axis indicator can hide.

### Art. 10 — Data and data governance

Mapped metrics: the F1 family (multi-class) or EO / DP / CAL (binary),
plus the per-class F1 gap and the intersectional gaps. The intersectional
mapping is the answer to Art. 10(2)(g): "examine the data sets in view of
possible biases [...] that are likely to affect [...] in particular where
data outputs influence inputs for future operations". An axis that looks
clean in isolation but compounds with another axis is exactly what
Art. 10(2)(g) asks providers to detect.

### Art. 13 — Transparency / provision of information to deployers (NEW in S7)

The model card carries the contextual information Art. 13 requires:
system identification, intended purpose, instructions for use, known
limitations, residual risks, oversight summary, logging policy. The CLI
embeds the model-card YAML under the Art. 13 sub-block so the evidence
pack is self-contained from an auditor's standpoint.

Template: [`src/fmm_fairness/templates/ai_act/model-card.yaml`](../src/fmm_fairness/templates/ai_act/model-card.yaml).

### Art. 14 — Human oversight (consolidated in S7)

Mapped metrics: `ai_vs_pooled_raters_kappa`, `cohen_kappa_matrix`,
`fleiss_kappa`, `krippendorff_alpha`.

The pooled-rater Cohen kappa is the headline Art. 14 metric: it places
the AI against the human-expert reference. The kappa matrix documents the
level of disagreement among the human raters themselves; that is the
floor of what the AI can be expected to inherit. The S7 update adds an
explicit `has_rater_evidence` boolean so an absent rater corpus is a flag,
not a silent omission.

Template: [`src/fmm_fairness/templates/ai_act/human-oversight.md`](../src/fmm_fairness/templates/ai_act/human-oversight.md).

### Art. 15 — Accuracy, robustness, cybersecurity

Mapped metrics: `inter_site_auc_variance`, `multi_class_auc_gap`,
`brier_score`, `hosmer_lemeshow`, `permutation_p_value`,
`minimum_detectable_effect`.

S6 added Brier + Hosmer-Lemeshow under Art. 15 to close the gap-reported,
calibration-silent audit finding. S7 keeps this mapping and additionally
cites the permutation p-value + MDE from S4 as the inferential support
for the observed inter-site gap.

### Art. 72 — Post-market monitoring (NEW in S7)

CSV schema template: [`src/fmm_fairness/templates/ai_act/post-market-monitoring.csv`](../src/fmm_fairness/templates/ai_act/post-market-monitoring.csv).
Procedure: [`src/fmm_fairness/templates/ai_act/post-market-monitoring.md`](../src/fmm_fairness/templates/ai_act/post-market-monitoring.md).

The Art. 72 obligation is operational, not numeric: the provider must
capture every inference, every reviewer disagreement, every change in the
population, and re-run the evaluation periodically. The CLI publishes the
CSV schema the operator should populate and the drift-detection
thresholds (overridable per deployment). A breach of any threshold
triggers an Art. 72 incident report.

Default drift thresholds:

| Signal | Default |
|---|---|
| weighted_f1_gap absolute change | 0.10 |
| Brier absolute change | 0.02 |
| ai_vs_pooled kappa drop | 0.10 |
| Hosmer-Lemeshow p-value floor | 0.01 |

The first signal compares against the operator-declared
`threshold_accept` in the model card. The remaining three are relative
to the reference evidence pack.

## 4. Backward compatibility

- Pre-S7 callers using `--manifest-mode ai-act` are unchanged. The block
  shape is preserved and `ai_act_full` is not emitted.
- Pre-S7 evidence packs do not have the `ai_act_full` key; downstream
  readers should treat its absence as "the operator was on a basic
  mapping".
- The CLI accepts both `--manifest-mode ai-act` and `--manifest-mode
  ai-act-full`; an unknown value still fails fast at argparse.

## 5. Template pack versioning

Templates ship at `src/fmm_fairness/templates/ai_act/`. The pack version is declared by
`fmm_fairness.ai_act_dossier.TEMPLATE_PACK_VERSION` and embedded into
every evidence pack under `ai_act_full.template_pack.version`. Bumps to
the pack are documented in the CHANGELOG.

Forking the templates is supported: pass `AiActFullConfig(template_pack_path=...)`
to the Python API to point the dossier at a forked template directory.
The CLI does not yet surface this knob (default templates are usually the
right answer); operators with bespoke regulatory wording can drive the
override from a thin Python wrapper.

## 6. PyYAML hard dependency

The Art. 13 model card uses YAML for human readability. PyYAML is added
as a runtime dependency so the loader is one well-tested implementation
across all installs; the prior fallback parser was insufficient for
list-of-mapping shapes used by `residual_risks` and `limitations`. The
extra cost is a small dependency that is already present in most Python
environments.

## 7. References

- Regulation (EU) 2024/1689 (the AI Act).
- ISO/IEC 42001:2023 — Information technology — Artificial intelligence
  management system.
- Mitchell, M. et al. 2019. *Model cards for model reporting.* FAT*.
  (Origin of the model-card format.)
- Liang, W. et al. 2022. *Advances, challenges and opportunities in
  creating data for trustworthy AI.* Nat. Mach. Intell. (Post-market
  monitoring data practice.)
