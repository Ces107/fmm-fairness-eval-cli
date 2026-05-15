# Example workflow — from predictions CSV to AI Act Art. 10 evidence pack

This walk-through shows the full path from a freshly trained dermatology-AI model to a regulator-ready evidence artifact. Total wall-clock: under five minutes once you have predictions.

## 0. Prerequisites

- Python 3.10+
- `pip install fmm-fairness-eval`
- A predictions CSV with the columns `y_true,y_pred,y_score,<your protected attributes>`.

If you don't have a CSV handy, generate the demo one:

```bash
python examples/synth_predictions.py > examples/predictions.csv
```

This simulates a 3-hospital dermatology setting where hospital A's model performs strongly, B medium, C poorly — the inter-hospital generalization gap the tool is built to surface.

---

## 1. Inspect the input

The CSV looks like this:

```
y_true,y_pred,y_score,site,sex,age_bucket
1,1,0.91,hospital_A,F,40-64
0,0,0.12,hospital_A,M,65+
1,0,0.42,hospital_C,F,0-39
...
```

**Required columns:**
- `y_true`: int {0, 1}
- `y_pred`: int {0, 1}  (your thresholded output — whatever threshold you would deploy at)
- `y_score`: float [0, 1]  (raw model probability)

**Declared protected attributes**: any column you pass via `--protected-attrs`. The tool does not auto-detect them — silent inference is itself a bias risk.

---

## 2. Run the evaluation

```bash
fmm-fairness evaluate examples/predictions.csv \
    --protected-attrs site,sex,age_bucket \
    --site-attribute site \
    --manifest-mode ai-act \
    --output dermatology-fairness-report/
```

Output:
```
OK: wrote evidence pack to dermatology-fairness-report/
  - dermatology-fairness-report/fairness-report.md  (sha256=a7b261ecd562...)
  - dermatology-fairness-report/fairness-evidence.json  (sha256=80898847ec9a...)
  - dermatology-fairness-report/audit.sha256
```

---

## 3. Read the human-readable report

`fairness-report.md` opens with a one-glance composite, then breaks down by protected attribute. A representative snippet:

```markdown
# Fairness evaluation report

- Tool: fmm-fairness-eval v0.1.0
- Generated (UTC): 2026-05-14T...Z
- Sample count: 900
- Protected attributes: site, sex, age_bucket

## SaMD composite fairness score
- Score: 0.7321  (1.0 = perfectly fair, 0.0 = maximally unfair)
- Components: site_term=0.3210, EO=0.1750, DP=0.0530, CAL=0.0410

## Inter-site AUC
- Variance (max-min): 0.025
  - hospital_A (n=300): AUC=0.94
  - hospital_B (n=300): AUC=0.86
  - hospital_C (n=300): AUC=0.71
```

Two readings of this output:
- **Headline**: composite is 0.73. Not a green light.
- **Diagnostic**: the `site_term` dominates the deduction; hospital C is the leak. The remediation is not "retrain with reweighting" — it is "investigate hospital-C data distribution" (probably a stain-protocol or capture-device shift if this is dermatology, the modal cause).

---

## 4. The machine-readable evidence pack

`fairness-evidence.json` (stable, sorted-key, deterministic) is the artifact you cite in an AI Act technical file. With `--manifest-mode ai-act`, the JSON includes a `regulatory_mapping` block:

```json
"regulatory_mapping": {
  "framework": "EU AI Act (Regulation 2024/1689)",
  "articles": [
    {
      "article": "Art. 10",
      "title": "Data and data governance",
      "mapped_metrics": ["equal_opportunity_gap", "demographic_parity_gap", "calibration_gap"],
      "note": "Per-attribute breakdown evidences Art. 10(2)(f-g) examination of biases and shortcomings."
    },
    ...
  ]
}
```

---

## 5. Drop into the AI Act technical file

The AI Act Annex IV (technical documentation) requires, among other things:
- (2)(b) the design specifications, including the rationale of the design choices...
- (2)(d) information about the data sets used, their provenance, scope, and main characteristics...
- (2)(g) the validation and testing procedures, including information about validation data, and the main metrics used to measure accuracy, robustness, and compliance...

Where this evidence pack lands:

| Annex IV section | Evidence file from this tool |
|---|---|
| 2(d) — data provenance & bias examination | `fairness-evidence.json` → `per_attribute_metrics.*` |
| 2(g) — validation metrics | `fairness-evidence.json` → `inter_site` + `samd_fairness_score` |
| 2(g) — robustness | `fairness-evidence.json` → `inter_site_auc_variance` |
| Art. 9 — risk management | composite score + components are the bias-residual-risk evidence |
| Art. 10(2)(f-g) — biases and shortcomings | per-attribute gap report |

Pin the `audit.sha256` line in your QMS change-control entry so the evidence is tamper-evident.

---

## 6. CI integration (preview — full hosted version in Phase 2)

You can run this in any CI that has Python. Example GitHub Actions step:

```yaml
- name: Fairness evaluation
  run: |
    pip install fmm-fairness-eval
    fmm-fairness evaluate artifacts/predictions.csv \
        --protected-attrs site,sex,age_bucket \
        --manifest-mode ai-act \
        --output fairness-report/
- uses: actions/upload-artifact@v4
  with:
    name: fairness-evidence
    path: fairness-report/
```

Add a step that fails the build if the composite drops below a threshold:

```yaml
- name: Enforce fairness floor
  run: |
    python -c "
    import json, sys
    e = json.load(open('fairness-report/fairness-evidence.json'))
    score = e['samd_fairness_score']['samd_fairness_score']
    print(f'samd_fairness_score = {score:.3f}')
    sys.exit(0 if score >= 0.80 else 1)
    "
```

That's the entire workflow.
