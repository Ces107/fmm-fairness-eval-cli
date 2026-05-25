# Art. 72 Post-market-monitoring collection pattern

EU AI Act Article 72 requires high-risk AI providers to "establish and
document a post-market-monitoring system [...] proportionate to the nature
of the AI technologies and the risks of the high-risk AI system". This
template defines the minimal CSV schema and the collection cadence the CLI
expects on the next evaluation cycle.

## CSV schema (`post-market-monitoring.csv`)

| Column | Type | Required | Description |
|---|---|---|---|
| `sample_id` | string | Y | Unique opaque identifier per inference. Pseudonymised, never PHI. |
| `site` | string | Y | Site / hospital / centre identifier. Same vocabulary as the training-time `site` column. |
| `timestamp_utc` | ISO 8601 | Y | When the inference happened. UTC. |
| `y_true` | int | (eventually) | Ground-truth class once available (post-confirmation). Empty when not yet known. |
| `y_pred` | int | Y | Class the AI predicted (argmax over score columns). |
| `y_score_0..y_score_{K-1}` | float | Y | Per-class probability emitted by the AI. |
| `reviewer_id` | string | (when reviewed) | Pseudonymised reviewer identifier. Empty when no human reviewed the case. |
| `reviewer_label` | int | (when reviewed) | The class the reviewer would have assigned. May equal `y_pred` (confirm) or differ (override). |
| `confidence_flag` | enum | Y | `high`, `medium`, `low` — derived from max(score). Operator sets the cut-offs. |
| `subgroup_flag` | string | N | Free-form tag for subpopulation. Used downstream to detect performance drift in declared subgroups. |
| `decision_outcome` | enum | Y | `confirmed`, `overruled`, `pending`, `escalated`. |
| `decision_modified_by_human` | bool | Y | True if `decision_outcome` was not `confirmed`. |
| `notes` | string | N | Free-form. Encryption / pseudonymisation per the provider's DPIA. |

## Collection cadence

- **Real-time append**: every inference writes one row at the time of
  inference. `y_true` is empty at this stage.
- **Backfill cadence**: once ground truth is available (clinical follow-up,
  pathologist confirmation, etc.), the row is updated in place with
  `y_true`. The CLI tolerates rows with empty `y_true` and excludes them
  from re-evaluation; rows with `y_true` populated are eligible for the
  next monitoring cycle.
- **Cycle cadence**: depending on risk tier and inference volume, the
  provider re-runs `fmm-fairness evaluate` on the appended CSV monthly,
  quarterly, or per regulatory request. The pack is appended to the
  technical-documentation file (Annex IV §7).

## Drift detection (what to look for)

The next evaluation cycle compares the freshly-evaluated CSV against the
reference evidence pack stored under `evidence_pack_path` in the model
card. Three drift signals trigger Art. 72 incident escalation:

1. **Inter-site gap increase**: the new `weighted_f1_gap` exceeds the
   reference + 2 × bootstrap SE (or the model-card `threshold_accept`),
   whichever is tighter.
2. **Calibration drift**: the new per-site Brier exceeds the reference
   Brier by more than 0.02 absolute, or any per-class Hosmer-Lemeshow
   p-value drops below 0.01.
3. **Oversight-disagreement drift**: `ai_vs_pooled_raters_kappa` drops by
   more than 0.1 absolute (Landis-Koch one tier) versus the reference.

## Privacy and PHI

Nothing in this schema is PHI. The provider's DPIA must establish how
`sample_id`, `reviewer_id`, and `subgroup_flag` are constructed so that
re-identification risk is acceptable for the deployment.

## Coupling to the CLI

When the provider runs `fmm-fairness evaluate --manifest-mode ai-act-full
--predictions post-market-monitoring.csv [...]` on the appended CSV, the
CLI produces a fresh evidence pack with the same shape as the reference
pack. The provider then diffs the two packs against the drift criteria
above and files an Art. 72 incident report if any criterion is breached.

The CSV schema is intentionally a subset of the predictions-CSV schema the
CLI already consumes, plus the operational columns
(`reviewer_id`, `confidence_flag`, `decision_outcome`,
`decision_modified_by_human`). The CLI ignores unknown columns; pre-S7
callers pass through unchanged.
