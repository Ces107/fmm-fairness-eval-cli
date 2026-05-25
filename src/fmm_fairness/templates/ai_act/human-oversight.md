# Art. 14 Human-oversight procedure template

EU AI Act Article 14 requires that "high-risk AI systems shall be designed
and developed in such a way [...] that they can be effectively overseen by
natural persons during the period in which they are in use". This template
formalises how a provider documents that oversight loop in pairing with
the numeric evidence the CLI produces.

The CLI surfaces the headline number for this article: the
`ai_vs_pooled_raters_kappa` Cohen kappa (with BCa CI) between the system
under evaluation and the per-item majority vote of the declared human
raters (`--rater-cols`). The kappa matrix and the Fleiss / Krippendorff
statistics surrounding it document the level of disagreement among the
human experts themselves, which is the floor of what the AI can be
expected to inherit.

## Oversight modes

Pick one and document it in the model card under `oversight_summary`:

- **Advisory** — the AI output is shown to a human reviewer who then makes
  the decision. The system is not the deciding agent. Lowest-risk
  oversight mode; appropriate when the AI is an assistant.

- **Mandatory second read** — the AI is run independently, the human is
  run independently, and disagreements are escalated to a third reviewer.
  This is the strongest oversight model for clinical-decision-support
  use cases and the one most defensible under Art. 14.

- **Autonomous-with-audit** — the AI acts but every k-th decision is
  audited by a human. This is appropriate for low-risk decisions at high
  volume; it is not a defensible substitute for the previous two modes in
  high-risk SaMD deployments.

## What the kappa numbers mean here

- Kappa `> 0.81` (Landis & Koch "almost perfect") — the AI behaves like
  a human-equivalent rater; Art. 14 oversight can be light (audit).
- Kappa `0.61 - 0.80` ("substantial") — the AI is a useful adjunct; the
  default for medical AI screening tools.
- Kappa `0.41 - 0.60` ("moderate") — the AI's residual disagreement is
  meaningful; Art. 14 oversight should be mandatory-second-read.
- Kappa `< 0.41` ("fair" or worse) — do not deploy without senior-clinician
  oversight on every decision.

## Disagreement-zone documentation

For each oversight mode the provider must declare:

1. **Trigger** — which AI output triggers a human review? Example:
   confidence below `--score-threshold`, prediction in a flagged class,
   patient in a flagged subgroup.
2. **Reviewer pool** — who can perform the review? Document the minimum
   qualification.
3. **Escalation rule** — what happens when the reviewer disagrees with the
   AI? Document the next step.
4. **Feedback loop** — disagreements feed back into the Art. 72
   post-market-monitoring CSV. The CLI's next run on the appended CSV
   re-computes kappa with the additional reviewer labels.
5. **Stop-the-line authority** — under what conditions does a human pause
   the system? Document the threshold and the procedure.

## Pairing with the evidence pack

When `--manifest-mode ai-act-full` is set, the evidence pack carries an
`ai_act_full.art_14` block referencing:

- `ai_vs_pooled_raters_kappa` (overall and per-site)
- `cohen_kappa_matrix` (pairwise human-vs-human and human-vs-AI)
- `fleiss_kappa` (single-number agreement floor)
- The oversight-mode declared in the model card

Together these are the numeric evidence supporting whatever oversight
procedure the provider declares in this template.
