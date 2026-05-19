"""Generate a synthetic dermatology-style predictions CSV for demos.

Usage:
    python examples/synth_predictions.py > examples/predictions.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

RNG_SEED = 20260514


def make_predictions(n_per_site: int = 300, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sites = ["hospital_A", "hospital_B", "hospital_C"]
    sex_levels = ["F", "M"]
    age_buckets = ["0-39", "40-64", "65+"]
    rows = []
    for s_idx, site in enumerate(sites):
        # Stronger model at site A, weaker at site C — mimics inter-hospital generalization gap
        pos_alpha = [9, 7, 4][s_idx]
        pos_beta = [2, 3, 5][s_idx]
        n = n_per_site
        y_true = rng.integers(0, 2, size=n)
        y_score = np.where(y_true == 1,
                           rng.beta(pos_alpha, pos_beta, size=n),
                           rng.beta(2, 9, size=n))
        y_pred = (y_score >= 0.5).astype(int)
        sex = rng.choice(sex_levels, size=n)
        age = rng.choice(age_buckets, size=n, p=[0.3, 0.4, 0.3])
        for yt, ys, yp, sx, ag in zip(y_true, y_score, y_pred, sex, age, strict=False):
            rows.append({
                "y_true": int(yt),
                "y_pred": int(yp),
                "y_score": round(float(ys), 4),
                "site": site,
                "sex": sx,
                "age_bucket": ag,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_predictions()
    df.to_csv(sys.stdout, index=False)
