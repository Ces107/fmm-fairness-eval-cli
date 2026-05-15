# Kill-gate -- fmm-fairness-eval-cli

**Set:** 2026-05-15.
**Day-0 = day of GitHub-release / PyPI / Zenodo publication.**
**Decision day = Day-30.** Secondary gate at d+60 for consulting inquiries.

This venture has a slower validation curve than typical OSS tools because the audience is academic / regulatory / industry-research, not general developer. The kill-gate reflects that.

---

## Pass conditions (any ONE = continue to Phase 2 / consulting upsell)

| Metric | Threshold | How measured |
|---|---|---|
| GitHub stars | ≥ 40 by d+30 | Public counter on repo |
| `pip install fmm-fairness-eval` per pypistats `last_week` | ≥ 20 by d+30 | https://pypistats.org |
| Real-affiliation issues (non-student) | ≥ 2 by d+30 | GitHub profile bio + employer link |
| Inbound email re: consulting / SaMD-team-help | ≥ 1 by d+30 | Mailbox |
| Citation in arxiv / OSF preprint within 90d | ≥ 1 by d+90 | Zenodo + Google Scholar |
| Consulting inquiry (€60-100/h SaMD-team-help) | ≥ 1 by d+60 | Mailbox |

**Hitting any one = green.**

---

## Fail conditions (ALL of the below ⇒ kill)

- < 10 stars by d+30
- 0 real-affiliation issues
- 0 inbound consulting emails by d+60
- 0 citations within 90d

If all hit, archive repo, post-mortem, retain code for personal portfolio (it represents the principal's TFG-area expertise in a recoverable form).

---

## Yellow zone

1 metric near threshold, others below half: escalate by writing one technical post on `dev.to` or `arxiv.org` (cs.CY or cs.LG) tied to the `samd_fairness_score` definition. Wait additional 30d. Re-evaluate at d+90.

---

## Phase 2 conditions

On green pass:
1. Stand up hosted "fairness CI on every commit" Stripe product, €99-149/mo for SaMD teams.
2. Publish a short white-paper expanding the `samd_fairness_score` derivation with retrospective on real (anonymised) cohorts.
3. Offer 1-2 consulting engagements at €80-100/h to fairness-troubled SaMD teams who reached out.
