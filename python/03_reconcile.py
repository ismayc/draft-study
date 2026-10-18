"""Verify the R and Python draft-model implementations agree.

Both sides rebuild the modeling frame independently from the same cached
inputs — Torvik CSV, DraftHistory, NBA minutes — so this checks the joins,
the name normalization, the feature engineering, AND the model fit.

Run:  python python/03_reconcile.py   (after 02_model.py and R/02_model.R)
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

TOL = 1e-6

TABLES = [
    ("coefficients_m0.csv", "coefficients_m0_r.csv", ["term"], ["coef"]),
    ("coefficients_m1.csv", "coefficients_m1_r.csv", ["term"], ["coef"]),
    ("metrics.csv", "metrics_r.csv", ["model"],
     ["train_auc", "test_auc", "train_logloss", "test_logloss"]),
    ("calibration_test.csv", "calibration_test_r.csv", ["quintile"],
     ["n", "mean_pred", "actual"]),
    ("pick_curve.csv", "pick_curve_r.csv", ["pick"], ["p_rotation", "coin_flip"]),
]


def compare(py_name: str, r_name: str, keys: list[str],
            cols: list[str]) -> tuple[bool, str]:
    py = pl.read_csv(OUT / py_name, null_values=["NA"])
    r = pl.read_csv(OUT / r_name, null_values=["NA"])
    merged = py.join(r, on=keys, how="full", suffix="_r", coalesce=True)
    if merged.height != py.height or py.height != r.height:
        return False, (f"row counts differ: py={py.height} r={r.height} "
                       f"joined={merged.height}")
    worst = 0.0
    for c in cols:
        d = (merged[c] - merged[f"{c}_r"]).abs().max()
        if d is None:
            return False, f"column {c}: null diff"
        worst = max(worst, d)
    return worst <= TOL, f"{py.height} rows, max abs diff {worst:.2e}"


def main() -> int:
    print(f"Reconciling R and Python draft-model implementations (tol {TOL:g})\n")
    all_ok = True
    for py_name, r_name, keys, cols in TABLES:
        ok, msg = compare(py_name, r_name, keys, cols)
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {py_name:24s} {msg}")
    print("\n" + ("ALL CHECKS PASS" if all_ok else "MISMATCHES FOUND"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
