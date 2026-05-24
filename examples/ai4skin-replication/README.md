# AI4SkIN multi-site fairness — golden replication example

This example reproduces the **inter-site weighted F1 gap** published in
Pereiro 2026, *Patología computacional aplicada a la clasificación de
tumores fusocelulares con MIL y modelos fundacionales* (TFG, Universitat
Politècnica de València), using only data that can be reconstructed
from the published thesis figures.

It is the canonical "does the CLI produce the right numbers on a
realistic SaMD multi-site eval?" smoke test.

## What it ships

```
ai4skin-replication/
├── confusion_matrices.json  # the published HUSC + HCUV CMs, as JSON
├── build_dataset.py         # expands the CMs into a row-per-sample CSV
├── predictions.csv          # 157 rows, regenerable from build_dataset.py
├── raters.csv               # 10-rater synthetic ratings
├── replicate.ipynb          # runs the CLI, prints + asserts headline numbers
└── README.md                # you are here
```

`predictions.csv` and `raters.csv` are committed for convenience but
both are byte-deterministic given `--seed 20260524`; you can always
regenerate them with:

```bash
python build_dataset.py
```

## Headline numbers reproduced (BCa, 2 000 bootstrap + 2 000 permutation)

| Metric                       | HCUV (n=41) | HUSC (n=116) | Gap     | BCa CI95         | Perm. p  | MDE @ 80 % |
|------------------------------|------------:|-------------:|--------:|------------------|----------|-----------:|
| Weighted F1 gap (site)       |     0.7567  |      0.9224  | 0.1657  | [0.040, 0.322]   | 0.0085   |     0.2015 |
| Macro F1 gap (site)          |     0.7227  |      0.9010  | 0.1783  | [0.018, 0.335]   | 0.0165   |     0.2406 |
| AI vs pooled-raters Cohen κ  |        —    |         —    | 0.8362  | [0.773, 0.899]   | —        |          — |

These match Table 6 of the TFG (`metrics_center_comparison`):

```
HCUV  Macro F1 = 0.723   F1 ponderado = 0.757
HUSC  Macro F1 = 0.901   F1 ponderado = 0.922
```

## A note on the "0.241 headline" cited in the thesis text

The thesis prose and `GLOBAL_F1BYCENTER.png` cite a weighted F1 gap of
**0.241** ( HUSC = 0.931, HCUV = 0.690 ). The thesis also publishes the
per-site confusion matrices (`bcmHUCV.png`, `blue_confusion_matrix_HCUV.png`)
which, recomputed cell-by-cell, give a weighted F1 gap of **0.1657**
( HUSC = 0.9224, HCUV = 0.7567 ) — i.e. Table 6, not the abstract.

The two numbers refer to different metric extractions on the same
underlying model. This replication targets **the figure that can be
audited against the published confusion matrices**, because that is the
one a third-party reproducibility check can rebuild from disclosed
artefacts. If you need the 0.241 figure for a citation, it is the
abstract number; if you need a number that lines up with the per-cell
confusion matrix, it is 0.1657.

The CLI emits both gap-class metrics (weighted, macro, per-class)
and a permutation p-value, so the substantive finding ("inter-site
fairness gap is non-zero, p ≈ 0.008") survives both choices.

## How to run

From the repo root:

```bash
# 1. (Re)generate the CSVs deterministically
python examples/ai4skin-replication/build_dataset.py

# 2. Run the fairness CLI (the same invocation used in the notebook + CI)
python -m fmm_fairness.cli evaluate \
    examples/ai4skin-replication/predictions.csv \
    --protected-attrs site \
    --site-attribute site \
    --rater-cols doc1,doc2,doc3,doc4,doc5,doc6,doc7,doc8,doc9,doc10 \
    --bootstrap-method bca \
    --bootstrap-iters 2000 \
    --permutation-iters 2000 \
    --output examples/ai4skin-replication/out

# 3. Or open the notebook
jupyter nbconvert --to notebook --execute \
    examples/ai4skin-replication/replicate.ipynb
```

The CI job in `.github/workflows/test.yml` runs step 3 in nb-execute mode
and asserts the headline gap stays within ±0.005 of 0.1657 (the
audit-against-published-CMs target).

## Disclaimers

- `predictions.csv` is **not** patient-level prediction data. It is the
  minimal row-per-sample expansion of the published confusion matrices
  with score columns synthesised to peak at `y_pred`. Use it only as a
  CLI-shape fixture; do not draw clinical or scientific conclusions
  from the score column distributions.
- `raters.csv` is fully synthetic. The 10-rater agreement is
  parameterised to land at a published-plausible AI-vs-pooled κ
  range, not to reconstruct any real rater. Do not cite individual
  doc<i> columns as anything other than "a synthetic 82%-accuracy
  rater whose disagreements follow the model's own error pattern".
- Class names are the six fusocelular entities published in the thesis
  (Leiomioma, Leiomiosarcoma, Dermatofibroma, Dermatofibrosarcoma,
  Melanoma fusocelular, Fibroxantoma atípico). All TFG data is built
  on the public AI4SkIN dataset; no PHI passes through this example.
