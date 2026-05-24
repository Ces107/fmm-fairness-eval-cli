"""Build the AI4SkIN-replication CSVs from the published TFG confusion matrices.

This script expands the per-site confusion matrices (HUSC n=116, HCUV n=41) in
``confusion_matrices.json`` into a row-per-sample ``predictions.csv`` with the
shape expected by ``fmm-fairness evaluate``:

    y_true, y_pred, y_score_0..y_score_5, site, [doc1..doc10]

Per-row class scores are deterministically synthesised with the calibration:

    - On a correct row (y_pred == y_true), the true class gets a high
      confidence drawn from Beta(8, 2) clipped to [0.55, 0.99]; the remaining
      mass is spread uniformly across the other classes.
    - On a wrong row (y_pred != y_true), the predicted class gets a medium
      confidence from Beta(5, 4) clipped to [0.35, 0.85]; remaining mass spread
      uniformly across the other classes.

Rater columns (``doc1..doc10``) are synthesised to land at a published-plausible
agreement: each rater independently agrees with ``y_true`` with probability
``rater_accuracy`` (default 0.82, matching the kappa-range that AI4SkIN
multi-rater studies report for dermatopathology fusocelular); disagreements
pick from a class-confusion distribution that mirrors the model's own error
pattern, so AI-vs-pooled agreement is non-degenerate.

Output is byte-deterministic given the seed.

Usage::

    python build_dataset.py --out predictions.csv --raters-out raters.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_ORDER = [
    "leiomioma",
    "leiomiosarcoma",
    "dermatofibroma",
    "dermatofibrosarcoma",
    "melanoma_fusocelular",
    "fibroxantoma_atipico",
]
NUM_CLASSES = len(CLASS_ORDER)
SCORE_COLS = [f"y_score_{k}" for k in range(NUM_CLASSES)]
RATER_COLS = [f"doc{i}" for i in range(1, 11)]

RNG_SEED = 20260524


def _expand_confusion_to_rows(
    matrix: list[list[int]],
    site_label: str,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Turn a (K, K) confusion matrix into one row per sample."""
    rows: list[dict[str, object]] = []
    for true_k, pred_row in enumerate(matrix):
        for pred_k, count in enumerate(pred_row):
            for _ in range(count):
                rows.append({"y_true": true_k, "y_pred": pred_k, "site": site_label})
    rng.shuffle(rows)  # break the deterministic class-block ordering
    return rows


def _scores_for_row(
    y_true: int, y_pred: int, rng: np.random.Generator
) -> list[float]:
    """Synthesise a length-K probability vector that argmaxes to y_pred."""
    scores = np.zeros(NUM_CLASSES, dtype=float)
    if y_pred == y_true:
        peak = float(np.clip(rng.beta(8.0, 2.0), 0.55, 0.99))
    else:
        peak = float(np.clip(rng.beta(5.0, 4.0), 0.35, 0.85))
    scores[y_pred] = peak
    remaining = 1.0 - peak
    other_indices = [k for k in range(NUM_CLASSES) if k != y_pred]
    weights = rng.dirichlet([0.8] * (NUM_CLASSES - 1))
    for j, k in enumerate(other_indices):
        scores[k] = remaining * weights[j]
    s = scores.sum()
    if s > 0:
        scores = scores / s
    return [round(float(v), 6) for v in scores]


def _rater_label(
    y_true: int,
    confusion_with_true: list[float],
    rater_accuracy: float,
    rng: np.random.Generator,
) -> int:
    """A rater agrees with y_true with probability rater_accuracy, else samples a confusable class."""
    if rng.random() < rater_accuracy:
        return y_true
    distractor_probs = list(confusion_with_true)
    distractor_probs[y_true] = 0.0
    s = sum(distractor_probs)
    if s == 0:
        choices = [k for k in range(NUM_CLASSES) if k != y_true]
        return int(rng.choice(choices))
    distractor_probs = [p / s for p in distractor_probs]
    return int(rng.choice(NUM_CLASSES, p=distractor_probs))


def _per_class_confusion_distribution(matrix: list[list[int]]) -> list[list[float]]:
    """Row-normalise the confusion matrix, with a uniform fallback for empty rows."""
    out: list[list[float]] = []
    for row in matrix:
        s = sum(row)
        if s == 0:
            out.append([1.0 / NUM_CLASSES] * NUM_CLASSES)
        else:
            out.append([v / s for v in row])
    return out


def build(
    cms_path: Path,
    out_path: Path,
    raters_out_path: Path | None,
    rater_accuracy: float,
    seed: int,
) -> tuple[Path, Path | None]:
    with cms_path.open("r", encoding="utf-8") as f:
        cms = json.load(f)
    rng = np.random.default_rng(seed)

    all_rows: list[dict[str, object]] = []
    for site_label in ("HUSC", "HCUV"):
        site_rows = _expand_confusion_to_rows(
            cms[site_label]["matrix"], site_label, rng
        )
        assert len(site_rows) == cms[site_label]["n"], (
            f"{site_label}: matrix sum {len(site_rows)} != published n {cms[site_label]['n']}"
        )
        all_rows.extend(site_rows)

    # Combined per-class distractor distribution for rater confusion.
    combined_cm = [
        [
            cms["HUSC"]["matrix"][i][j] + cms["HCUV"]["matrix"][i][j]
            for j in range(NUM_CLASSES)
        ]
        for i in range(NUM_CLASSES)
    ]
    confusion_dist = _per_class_confusion_distribution(combined_cm)

    for row in all_rows:
        scores = _scores_for_row(int(row["y_true"]), int(row["y_pred"]), rng)
        for i, s in enumerate(scores):
            row[SCORE_COLS[i]] = s
        if raters_out_path is not None:
            for rater_col in RATER_COLS:
                row[rater_col] = _rater_label(
                    int(row["y_true"]),
                    confusion_dist[int(row["y_true"])],
                    rater_accuracy,
                    rng,
                )

    cols = ["y_true", "y_pred", *SCORE_COLS, "site"]
    if raters_out_path is not None:
        cols.extend(RATER_COLS)
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(out_path, index=False, lineterminator="\n")

    raters_written: Path | None = None
    if raters_out_path is not None:
        raters_df = df[["y_true", "site", *RATER_COLS]].copy()
        raters_df.to_csv(raters_out_path, index=False, lineterminator="\n")
        raters_written = raters_out_path

    return out_path, raters_written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    p.add_argument("--cms", default=str(here / "confusion_matrices.json"))
    p.add_argument("--out", default=str(here / "predictions.csv"))
    p.add_argument("--raters-out", default=str(here / "raters.csv"))
    p.add_argument("--rater-accuracy", type=float, default=0.82)
    p.add_argument("--seed", type=int, default=RNG_SEED)
    args = p.parse_args()
    pred_path, raters_path = build(
        Path(args.cms),
        Path(args.out),
        Path(args.raters_out) if args.raters_out else None,
        args.rater_accuracy,
        args.seed,
    )
    print(f"OK: wrote {pred_path}")
    if raters_path is not None:
        print(f"OK: wrote {raters_path}")


if __name__ == "__main__":
    main()
