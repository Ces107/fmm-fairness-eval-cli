# Show HN draft for fmm-fairness-eval v0.2.0a11

**Status:** Operator-prepared, awaits principal sign-off and personal-voice rewrite before posting under `Ces107` on Hacker News.

**Posting account:** `Ces107` (principal real identity). Per CLAUDE.md §12 still gated: "Communications sent under principal's real identity". Cannot be posted by operator.

**When to post:** Tuesday or Wednesday morning EU timezone for max HN visibility window (08:00-10:00 UTC). Today 2026-05-27 is a Wednesday, so the window for the next acceptable post is either today afternoon (~18:00-19:00 UTC for US morning) or next Tuesday 2026-06-02.

**Pre-flight check before submitting:**

1. Stripe payment link MUST be live in README pricing section (currently says "wiring 2026-05-21"). Either fix the link or rewrite the line. HN commenters will flag a dead pricing link in 30 seconds.
2. Tag v0.2.0a11 is pushed (confirmed 2026-05-27).
3. PyPI live URL confirmed (2026-05-27).
4. README first paragraph rewritten without em-dashes (current draft has several).
5. Open the first-comment text in a HN tab BEFORE submitting so it can be posted within 30 seconds of submission.

---

## Title (limit 80 chars)

Two candidates, pick whichever lands cleaner after principal reads them aloud:

```
Show HN: fmm-fairness-eval, a CLI for inter-hospital bias in SaMD models
```

```
Show HN: I built a CLI that ships an EU AI Act fairness evidence pack
```

The first one leads with the technical narrow scope (inter-hospital bias, SaMD). The second leads with the regulatory outcome (Art. 10 evidence pack). Default to the first for an engineering audience; switch to the second if principal wants to land in HN-business-of-software discussions.

---

## First comment body (post within 30 seconds of submission)

```
Author here. Brief context on why this exists.

I wrote this for the failure mode my TFG ran into: a CONCH-embedding plus MIL classifier on the AI4SkIN dermatology cohort that scored F1=0.89 weighted but had a 0.19 gap between the two hospital sites in the data. That gap is not visible in any of FairLearn, AIF360, or Holistic AI by default, because none of them treat hospital site as a first-class protected attribute distinct from individual demographics.

What fmm-fairness-eval does:

1. Takes a predictions CSV that includes a site or hospital column plus the usual demographic columns.
2. Computes per-site AUC variance, weighted F1 gap by axis, intersectional gap with BCa confidence intervals and permutation p-values, calibration (Brier, Hosmer-Lemeshow), and a composite SaMD fairness score.
3. Emits a Markdown evidence pack with verbatim AI Act Art. 9 and Art. 10 citations, a JSON machine-readable pack, and a SHA-256 audit chain so the pack is tamper-evident the moment it leaves the pipeline.

Tech: pure Python 3.10+, no GPU dependency, MIT license, runs in a CI step or as a local pre-submission check. Tests cover 145 cases including statistical sanity (small-sample exclusion, permutation null preserved under random labels, BCa coverage). Ruff plus mypy strict clean.

Repo: https://github.com/Ces107/fmm-fairness-eval-cli
PyPI: https://pypi.org/project/fmm-fairness-eval/

This is alpha. The numbers are correct, the API is not frozen. If you are evaluating a SaMD against a multi-site cohort and want to try it on your own predictions CSV, the README has a 60-second walkthrough. Honest feedback on what fairness axes are missing, or what regulators in your jurisdiction would push back on, is the most useful thing right now.
```

Word count of first comment body: 312. Fits HN comfortably.

---

## Anticipated comment threads and prepared replies

These are operator-drafted so principal can fire one-line replies fast if they come up. Each is keyed to the predicted critique.

### "How is this different from FairLearn / AIF360?"

```
FairLearn and AIF360 are general-purpose ML fairness libraries. They give you the metrics. They do not give you a regulator-shaped evidence artifact and they do not treat hospital site as protected. fmm-fairness-eval is opinionated about the SaMD case: per-site is first-class, the composite score weights bias categories the way EU notified bodies actually rank them, and the output is a Markdown plus JSON evidence pack with cross-cited Art. 9 and Art. 10 references. You can wrap FairLearn around your SaMD evaluation and reach the same numbers; you cannot ship that to a notified body without another two weeks of evidence drafting.
```

### "Is this just compliance theater?"

```
The math is statistically real, not cosmetic. Permutation p-values, BCa confidence intervals, small-cell exclusion at n less than 20 with a documented threshold. If a regulator asks why your model has a 0.19 gap between sites, the report tells them and the audit chain proves the numbers were not edited after the fact. The compliance framing exists because that is what makes hospitals buy fairness work. The underlying numbers are the same numbers a careful ML researcher would compute.
```

### "EU AI Act is not finalized for SaMD yet."

```
Correct. The Regulation entered into force August 2024 with phased application; obligations for high-risk AI systems (which most SaMD is) apply from August 2026. This tool ships the evidence shape the Regulation already specifies in Art. 10 (data governance) and Art. 9 (risk management) regardless of guidance still in flight. The point is not to game an enforcement gap, it is to have the artifact ready the day the gap closes.
```

### "What is the licensing on the consulting tier?"

```
Consulting is a separate engagement, not part of the OSS license. The CLI is MIT, free forever. Hosted CI and consulting are fee-for-service. Pricing is in the README.
```

---

## Post-launch follow-up checklist (operator-tracked, principal-fired)

- d+0 (post-day): check star delta every 2h. Reply to legit technical comments within 1h.
- d+1: capture HN URL into `state/portfolio_ventures.yaml` `hn_id` field for fmm.
- d+1: post the same announcement on dev.to and r/MachineLearning if HN traction is greater than 30 points.
- d+3: write a follow-up README section "real-world cohorts that fmm-fairness-eval has been run against" if anyone runs it.
- d+7: outbound batch (7 personas at state/work/out/outbound-fmm-batch-001.md sha 559E1F0C...) goes out, referencing the HN thread.

---

## What this draft does NOT do (deliberately)

- Does not lie about traction. Says "alpha" honestly.
- Does not over-promise on regulatory acceptance. The product is evidence-shape compliance, not a certification.
- Does not include the consulting CTA in the first comment. The pitch is the tool, not the upsell.
- Does not name Laberit, Ascires, Juaneda, or any specific Laberit client. Off-limits per CLAUDE.md §3.
- Does not name the principal's network. Off-limits per principal directive 2026-05-05.
