# Monetization gate — fmm-fairness-eval

Per `skills/monetization-gate/SKILL.md`. As-of date: 2026-05-20 pre-Stripe-live.

## Section A: Visible revenue path

- [Y] README has a "Pricing" block in the first 10 lines.
  - Evidence: `README.md` top "## Pricing" block listing tier names + prices.
- [Y] README links to `pricing.md` in the repo with at least 2 tiers.
  - Evidence: `pricing.md` exists with 3 tier(s).
- [Y] At least one tier has an actual price in EUR.
  - Evidence: tiers listed in `pricing.md` with EUR prices.
- [Y] A demo / install / signup path exists.
  - Evidence: README install or CTA visible top-of-file.

## Section B: Payment instrument

- [PENDING] Stripe / payment-link live.
  - Status: **PENDING — Stripe live keys 2026-05-21**. Principal completes Stripe / Wise EUR business setup on 2026-05-21; SKU `fmm-fairness-ci` to be created then; README block `[Stripe Payment Link — wiring 2026-05-21]` to be replaced with the live link in a single follow-up edit.
- [PENDING] One payment link generated and €1 test transaction confirmed.
  - Blocked by item above.
- [Y] Invoicing template exists.
  - Evidence: `legal/invoice-template.md`.
- [Y] VAT handling decided: B2B EU reverse-charge / out-of-scope non-EU / deferred for ES.
  - Evidence: `pricing.md` §VAT, `legal/invoice-template.md` §VAT decision tree.

## Section C: Legal minimums

- [Y] LICENSE chosen (MIT for free tier source).
  - Evidence: `LICENSE` at repo root.
- [Y] Terms of Service draft exists.
  - Evidence: `legal/tos.md`.
- [Y] Privacy Policy draft exists.
  - Evidence: `legal/privacy.md`.
- [Y] GDPR section present.
  - Evidence: `legal/privacy.md` is fully GDPR-anchored.

## Section D: Outcome defined

- [Y] The outcome being sold is defined, not per-hour.
  - Outcome sold: a regulator-friendly fairness evidence pack (Markdown + JSON + SHA-256 chain) per model release, plus optional expert review hours for AI Act Art. 9 / Art. 10 dossier wording. Not sold by lines of code; sold by completed evidence pack and by reviewed hour.

## Section E: Kill-gate

- [Y] Kill-gate explicit: <5 stars + 0 commercial signal at D+14 → archive; revenue-target whiff at D+30 → archive.
  - Evidence: `kill-gate.md` (existing) plus this gate file.

## Overall verdict

- A: Y, B: **PENDING (Stripe live keys 2026-05-21)**, C: Y, D: Y, E: Y.
- Ship-block: no live payment instrument yet. All other gates green. One-edit delta tomorrow.

**As of:** 2026-05-20.
**Re-check trigger:** Stripe live keys land. Replace `[Stripe Payment Link — wiring 2026-05-21]` placeholders in README + `pricing.md`, then re-run this checklist and flip B to Y.
