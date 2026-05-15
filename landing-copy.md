# Landing copy -- fmm-fairness-eval-cli

For the future Carrd / static landing page.

---

## Hero

**Foundation-model fairness evidence, on every commit.**

A SaMD-specific CLI that emits the inter-hospital / inter-cohort fairness report regulators and reviewers actually ask for. Open-source. MIT-licensed. Runs on your machine.

[GitHub](https://github.com/<github-handle>/fmm-fairness-eval-cli) -- [Zenodo DOI](https://doi.org/...) -- [arxiv preprint](#)

---

## What it does

Feed it predictions + protected attributes. Get an AI Act Article 10 evidence pack out:

- Equal Opportunity gap per protected attribute.
- Demographic Parity gap.
- Calibration gap per subgroup.
- Inter-site AUC variance.
- A composite `samd_fairness_score` defined transparently at `docs/samd-fairness-score.md` (not a black box).
- A hash-chained `audit.sha256` so a reviewer can verify the report's integrity from the JSON alone.

---

## Why this exists

The principal's TFG (Universitat Politècnica de València) measured inter-hospital fairness on dermatology AI using foundation-model embeddings (CONCH) on the AI4SkIN dataset and found a 0.19 inter-hospital fairness gap on a model with F1 = 0.89 weighted. The pattern repeats across SaMD: average performance is good, but per-site / per-cohort fairness is the failure mode regulators (FDA SaMD AGT, EU AI Act Art. 10) actually focus on.

Existing fairness libraries (FairLearn, AIF360, Microsoft RAI Toolbox, Holistic AI) are general-purpose. They do not package the SaMD-specific evaluation workflow, do not cite the regulation per-metric, and do not emit a hash-chained evidence pack.

---

## For who

- SaMD product teams preparing AI Act Art. 10 documentation.
- Medical-imaging-AI researchers running multi-site validation.
- Notified Body conformity-assessment auditors who want a structured, traceable artifact instead of a free-form fairness section.
- Hospital data-science teams trying to demonstrate equitable model performance to their DPIA / clinical-governance committee.

---

## Pricing

- **OSS CLI** -- always free, MIT.
- **Consulting (later)** -- short engagement to help a SaMD team interpret their evidence pack and connect it to their Art. 10 docs. Email `plusultra.dev@proton.me`.
- **Hosted CI (later)** -- fairness eval on every commit, retained audit log, multi-cohort comparison. €99-149/mo when launched. Reserve below.

---

## Reserve early access

Form below collects email + role + organisation. No tracking, no selling, one-click unsubscribe.

(Form embed)

---

## Why us

The author (UPV biomedical engineering, TFG on fairness-aware SaMD; CONCH + MIL on AI4SkIN; 2 years EHR/PACS engineering at a Spanish healthtech) is in the niche by training, not by acquisition. The TFG citation is at https://riunet.upv.es/handle/10251/226903.
