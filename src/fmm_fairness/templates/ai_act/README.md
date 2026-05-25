# EU AI Act dossier templates

Bundled reference templates inside the `fmm_fairness` package, paired with
`fmm-fairness evaluate --manifest-mode ai-act-full`. The CLI emits the
numeric evidence (per-attribute gaps, BCa CIs, permutation p-values, Brier
scores,
Hosmer-Lemeshow, intersectional cross-products); these templates carry the
contextual artefacts a provider needs alongside the numbers to close an AI
Act technical-documentation file (Annex IV).

## Files

- `model-card.yaml` — Art. 13 transparency template. Pass to the CLI via
  `--model-card path/to/model-card.yaml` to embed its contents in the
  `ai_act_full` block of the evidence pack.
- `human-oversight.md` — Art. 14 oversight-procedure template. Cross-cites
  the inter-rater agreement statistics emitted by `--rater-cols`.
- `post-market-monitoring.csv` — Art. 72 post-market-monitoring schema:
  one row per ongoing production sample, with the columns the operator must
  capture and re-feed into the CLI on the next cycle.
- `post-market-monitoring.md` — collection-pattern guide for the CSV.

## Versioning policy

These templates are versioned independently of the CLI source. The
`templates/ai-act/VERSION` file (if present in a future release) declares
the template-pack version; the CLI logs the version it bundled in the
`ai_act_full.template_pack_version` field of the evidence pack.

## How to extend

The templates are deliberately small and additive. Add fields to
`model-card.yaml` freely; the CLI passes through unknown keys to the
evidence pack. Removing or renaming a documented field requires a major
version bump because downstream auditors may rely on its presence.

## Disclaimer

These templates are reasonable defaults aligned with the published text of
Regulation (EU) 2024/1689. They are not legal advice. The provider remains
responsible for the conformity-assessment process; the CLI documents the
numeric evidence and the dossier shape, not the legal sufficiency.
