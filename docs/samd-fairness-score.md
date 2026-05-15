# Appendix — `samd_fairness_score`: definition, justification, sensitivity

This document defends the composite metric `samd_fairness_score` shipped in `fmm_fairness.metrics.samd_fairness_score`. It is the only opinionated number the tool produces; everything else is a direct, well-defined statistical quantity. The composite exists because regulators and quality-management systems (QMS) need a single dashboard number, and a black-box score is worse than a transparent, defensible one.

---

## 1. Formula

Let `D` be the set of declared demographic protected attributes (e.g. `{sex, age_bucket}`). Let `s` be the site/hospital attribute (default name: `site`).

Define:

- `EO  = mean_{a ∈ D} equal_opportunity_gap(a)`
- `DP  = mean_{a ∈ D} demographic_parity_gap(a)`
- `CAL = mean_{a ∈ D} calibration_gap(a)`
- `SITE = min(1, 2 * sqrt(Var_{g ∈ s}(AUC(g))))`

Then:

```
samd_fairness_score = 1 - clip( w_site * SITE + w_eo * EO + w_dp * DP + w_cal * CAL, 0, 1 )
```

with default weights `w_site = 0.40, w_eo = 0.30, w_dp = 0.15, w_cal = 0.15` (sum to 1.0).

Range: `[0, 1]` where **1 = perfectly fair** (no measured gaps) and **0 = maximally unfair** (gaps saturate the unit interval).

The `2 * sqrt(...)` rescaling on `SITE` maps the realistic range of inter-site AUC standard deviation (≈ 0 to 0.5) onto `[0, 1]`. Empirically, an AUC standard deviation of 0.10 across 3–5 sites already corresponds to "the model is unsafe to deploy at sites it was not validated at"; that maps to `SITE = 0.2`, contributing `0.08` to the score deduction at default weights.

---

## 2. Why these four components

A SaMD fairness audit must answer four distinct regulator-facing questions:

1. **Does the model miss disease equally across protected groups?** → `equal_opportunity_gap`, the TPR-gap criterion of Hardt, Price, Srebro (NeurIPS 2016).
2. **Does the model treat groups symmetrically in selection rate?** → `demographic_parity_gap`, the classical statistical-parity criterion of Dwork et al. (ITCS 2012).
3. **Is the score trustworthy to the same degree across groups?** → `calibration_gap`, motivated by Pleiss et al. (NeurIPS 2017) on the incompatibility of calibration and TPR-equality.
4. **Does the model generalize across hospital boundaries?** → `inter_site_auc_variance`, the practical SaMD failure mode emphasized by Seyyed-Kalantari et al. (Nat. Med. 2021) and the FDA/IMDRF Good Machine Learning Practice (GMLP) guiding principles (2021, updated IMDRF January 2025).

Dropping any of the four loses a regulator-facing answer. Adding more (e.g. equalized odds, predictive parity) would either duplicate signal already in (1)+(2)+(3) or import the well-known impossibility-theorem trilemma (Chouldechova 2017, Kleinberg-Mullainathan-Raghavan 2017) into a single number, which is precisely what a defensible composite must avoid.

---

## 3. Why these weights

FDA GMLP (2021) and the EU AI Act (Art. 9 + Art. 10, 2024/1689) both place **multi-site representativeness and generalization** as a first-tier obligation. The IMDRF GMLP guiding principles (January 2025) call out "Representative Data in Clinical Studies" and explicitly flag patient populations not well-represented in training as a transparency obligation. The dermatology AI fairness literature (Roy et al. 2022 / FairDisCo and follow-ups; Daneshjou et al. 2022 *Sci. Adv.* on skin-tone disparities) consistently shows that under-representation translates more directly into TPR gaps than into selection-rate gaps. This justifies:

- **`w_site = 0.40`**: the regulator-priority component, the failure mode this tool is named for, and the one no other OSS fairness library packages as a first-class metric.
- **`w_eo = 0.30`**: TPR-gap is the medical-AI-relevant fairness criterion par excellence (Pierson et al. 2021, Seyyed-Kalantari et al. 2021). A missed cancer in one subgroup is the harm a regulator will write up.
- **`w_dp = 0.15`**: demographic parity is a less-clinically-meaningful criterion in medicine (prevalence rightly varies by subgroup), so it is downweighted but kept for symmetry with non-medical fairness frameworks.
- **`w_cal = 0.15`**: calibration matters most for thresholding and clinical-decision-support workflows; in raw classification it is secondary.

These are defaults, not commandments. Override via the `weights` argument or report the four components separately.

---

## 4. Sensitivity analysis

The composite is sensitive to weights, as any weighted sum is. Two notes:

- **Monotonicity is preserved.** For any non-negative weights summing to 1, increasing any underlying gap can only decrease the score. This is desirable.
- **A bias hidden in one component can be hidden in the composite.** If your model has a large EO gap but a tiny SITE gap, the default weighting will compress that signal. **Always read the components, not only the composite.** The tool reports both.

A worked example: suppose `EO = 0.20, DP = 0.05, CAL = 0.05, SITE = 0.10`. With default weights:
```
raw = 0.40·0.10 + 0.30·0.20 + 0.15·0.05 + 0.15·0.05 = 0.04 + 0.06 + 0.0075 + 0.0075 = 0.115
score = 1 - 0.115 = 0.885
```
A model scoring 0.885 has a real 20-percentage-point TPR disparity that should not be papered over by the headline number.

---

## 5. What this score is NOT

- **Not** an FDA-certified metric.
- **Not** a green-light to deploy. A perfect 1.0 on a poorly-chosen test set tells you nothing.
- **Not** a substitute for prospective multi-site validation.
- **Not** stable across redefinitions of the protected attributes. Declaring `age_bucket` with 5 levels vs 3 levels can move the score by 0.05 even on identical predictions.

The composite exists to answer the question "give me one number for the QMS dashboard" without misrepresenting the underlying evidence. Use it in conjunction with the per-component report, not in place of it.

---

## 6. References

- **Hardt, M., Price, E., Srebro, N.** (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS. https://arxiv.org/abs/1610.02413
- **Dwork, C., Hardt, M., Pitassi, T., Reingold, O., Zemel, R.** (2012). *Fairness Through Awareness.* ITCS.
- **Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., Weinberger, K.** (2017). *On Fairness and Calibration.* NeurIPS.
- **Chouldechova, A.** (2017). *Fair prediction with disparate impact.* Big Data 5(2):153-163.
- **Pierson, E., Cutler, D. M., Leskovec, J., Mullainathan, S., Obermeyer, Z.** (2021). *An algorithmic approach to reducing unexplained pain disparities in underserved populations.* Nat. Med. 27:136-140. https://doi.org/10.1038/s41591-020-01192-7
- **Seyyed-Kalantari, L., Zhang, H., McDermott, M. B. A., Chen, I. Y., Ghassemi, M.** (2021). *Underdiagnosis bias of AI algorithms applied to chest radiographs in under-served patient populations.* Nat. Med. 27:2176-2182. https://doi.org/10.1038/s41591-021-01595-0
- **Char, D. S., Shah, N. H., Magnus, D.** (2018). *Implementing Machine Learning in Health Care — Addressing Ethical Challenges.* NEJM 378:981-983. https://doi.org/10.1056/NEJMp1714229
- **Lu, M. Y., Chen, B., Williamson, D. F. K., et al.** (2024). *A visual-language foundation model for computational pathology.* Nat. Med. 30:863-874. https://doi.org/10.1038/s41591-024-02856-4
- **FDA, Health Canada, MHRA** (2021). *Good Machine Learning Practice for Medical Device Development: Guiding Principles.* https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles
- **IMDRF** (January 2025). *Good Machine Learning Practice for Medical Device Development: Guiding Principles* (final).
- **European Union** (2024). *Regulation (EU) 2024/1689 on Artificial Intelligence (AI Act).* Articles 9, 10, 15.
- **Du, S., Hers, B., Bayasi, N., Hamarneh, G., Garbi, R.** (2022). *FairDisCo: Fairer AI in Dermatology via Disentanglement Contrastive Learning.* ISIC Workshop. https://arxiv.org/abs/2208.10013
- **Pereiro, C.** (2024). *Foundation-model-based fairness evaluation in dermatology classification using the AI4SkIN dataset.* TFG, Universitat Politècnica de València. https://riunet.upv.es/handle/10251/226903
