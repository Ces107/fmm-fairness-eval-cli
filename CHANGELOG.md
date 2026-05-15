# Changelog

All notable changes to this project will be documented here.

## v0.1.0 -- 2026-05-15 (initial release)

- CLI `fmm-fairness evaluate` computes Equal Opportunity, Demographic Parity, Calibration, Inter-site AUC variance, and the composite `samd_fairness_score`.
- Emits human-readable `fairness-report.md` plus machine-readable `fairness-evidence.json`.
- SHA-256 audit chain over each report (file `audit.sha256` companion).
- `--manifest-mode ai-act` toggle cross-cites the evidence to EU AI Act Art. 9 + Art. 10.
- 14 unittest cases passing under Python 3.10/3.11/3.12.
- Documented `samd_fairness_score` composite at `docs/samd-fairness-score.md`.
- Example workflow at `docs/example-workflow.md` walks through synthetic dermatology-AI predictions to evidence-pack.
- Research-artifact framing accompanying the author's TFG on fairness-aware SaMD (https://riunet.upv.es/handle/10251/226903).

## Roadmap (v0.2 candidates, dependent on signal)

- Pre-built protected-attribute discovery heuristics.
- Bootstrap-based confidence intervals on `samd_fairness_score`.
- Calibration-by-subgroup with Hosmer-Lemeshow goodness-of-fit.
- Integration with MONAI Bundle outputs.
- Hosted CI mode (€99-149/mo) running fairness eval per PR.
