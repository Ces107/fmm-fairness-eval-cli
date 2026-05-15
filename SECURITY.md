# Security policy

## Supported versions

The latest released version is supported.

## Reporting a vulnerability

Please email `plusultra.dev@proton.me` (preferred) with a description of the issue and reproduction steps. Do not file public GitHub issues for security reports.

We aim to acknowledge within 7 days and to publish a fix or mitigation within 30 days for confirmed vulnerabilities. For coordinated disclosure timelines, please indicate any embargo you require.

## Data-handling note

This tool processes prediction CSVs that may contain sensitive demographic attributes (age, sex, ethnicity, site). The CLI never transmits data over the network. Audit artifacts (`fairness-evidence.json`, `audit.sha256`) contain only aggregate statistics, not row-level data. Operators are responsible for ensuring that any prediction CSV is itself produced from a lawfully-anonymised cohort prior to evaluation.
