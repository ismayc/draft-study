"""Draft model: does college production add information beyond draft position?

The draft pick itself is the scouting consensus — the strongest single
predictor of NBA outcomes and the right baseline to beat. So the study fits
two logistic models of "rotation player within three seasons":

  M0  log(pick) only                      (the scouts)
  M1  log(pick) + college production      (the scouts plus the numbers)

and asks how much the college numbers move out-of-sample discrimination and
log-loss on draft classes the model never saw (train 2011-2018, test
2019-2022). Outcome: >= 2,000 NBA minutes within three seasons of the draft
(a season of real rotation minutes), joined by NBA PERSON_ID — the only name
matching anywhere is Torvik <-> DraftHistory, and it is gated.

Same hand-rolled IRLS as ../shot-quality-study, mirrored in ../R/02_model.R.

Run: python python/02_model.py    (after 01_harvest.py)
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
FIG = ROOT / "figures"

TRAIN_YEARS = range(2011, 2019)
TEST_YEARS = range(2019, 2023)
ROTATION_MIN = 2000.0        # NBA minutes within 3 seasons of the draft
CLASS_NUM = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}

FEATURES = ["log_pick", "age", "ht_in", "obpm", "dbpm", "usg", "ts_per",
            "orb_per", "drb_per", "ast_per", "to_per", "blk_per", "stl_per",
            "three_a_pg", "ftr", "class_num"]

# Torvik registers formal names; the NBA registers the name the player goes
# by. The alias table is a committed CSV shared with R/02_model.R, one reason
# per row (names in normalized norm_name form). A row with a torvik_year also
# admits that one Torvik (name, year) row despite a blank pick column.
# Draftees who match nothing on purpose are in draft_join_allowlist.csv, and
# 05_join_gate.py fails on any other unmatched 2011-2022 college draftee.
ALIAS_FILE = DATA / "draft_name_aliases.csv"
ALLOWLIST_FILE = DATA / "draft_join_allowlist.csv"


def load_aliases() -> tuple[dict[str, str], set[str]]:
    """(torvik name -> draft-history name, {'name|year' keys admitted with a
    blank pick})."""
    t = pl.read_csv(ALIAS_FILE, infer_schema_length=0)
    aliases = dict(zip(t["torvik_name_n"], t["dh_name_n"]))
    admit = {f"{n}|{y}" for n, y in zip(t["torvik_name_n"], t["torvik_year"])
             if y}
    return aliases, admit


NICKNAME_ALIASES, ADMIT_BLANK_PICK = load_aliases()


def norm_name(name: str) -> str:
    """Fold case/diacritics, drop punctuation and generational suffixes so
    'Gary Trent Jr.' and 'Gary Trent' collide on purpose."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = "".join(c for c in s.lower() if c.isalpha() or c == " ")
    parts = [p for p in s.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


def ht_inches(ht: str) -> float | None:
    """'6-10' -> 82.0; None when unparseable."""
    try:
        feet, inches = ht.split("-")
        return float(int(feet) * 12 + int(inches))
    except (ValueError, AttributeError):
        return None


def age_at_draft(dob: str, draft_year: int) -> float | None:
    """Age in years on draft night (approximated as June 25 of draft year)."""
    try:
        y, m, d = (int(x) for x in dob.split("-"))
    except (ValueError, AttributeError):
        return None
    return (draft_year - y) + (6 - m) / 12 + (25 - d) / 365


def logistic_irls(X: np.ndarray, y: np.ndarray, *, ridge: float = 1e-8,
                  tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
    """Identical algorithm to shot-quality-study; see that study's docs."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = p * (1.0 - p)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(X @ beta)))


def auc(pred: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties — the same closed form
    in both languages, so it reconciles exactly."""
    order = np.argsort(pred, kind="stable")
    ranks = np.empty(len(pred))
    sorted_pred = pred[order]
    i = 0
    while i < len(pred):
        j = i
        while j + 1 < len(pred) and sorted_pred[j + 1] == sorted_pred[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return (float(ranks[y == 1].sum()) - n1 * (n1 + 1) / 2) / (n0 * n1)


def log_loss(pred: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(pred, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def quintile_calibration(pred: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    order = np.argsort(np.round(pred, 10), kind="stable")
    q = np.empty(len(pred), dtype=int)
    q[order] = (np.arange(len(pred)) * 5) // len(pred)
    return (pl.DataFrame({"quintile": q + 1, "pred": pred, "y": y})
            .group_by("quintile")
            .agg(n=pl.len(), mean_pred=pl.col("pred").mean(),
                 actual=pl.col("y").mean())
            .sort("quintile"))


def match_draftees() -> dict:
    """Join Torvik final college seasons to DraftHistory by normalized name.
    Returns the frames the gate and build_dataset share: `college` (every
    2011-2022 college draftee), `by_name`, `joined`, and `final`."""
    tv = pl.read_csv(DATA / "torvik_2011_2022.csv", infer_schema_length=0)
    tv = tv.with_columns(
        pl.col("year").cast(pl.Int64),
        raw_n=pl.col("player_name").map_elements(norm_name,
                                                 return_dtype=pl.String))
    # Torvik's pick column filters to drafted players; a few drafted players
    # have it blank, and the alias table admits those rows by (name, year).
    admit = (pl.col("raw_n") + "|" + pl.col("year").cast(pl.String)
             ).is_in(sorted(ADMIT_BLANK_PICK))
    picked = (tv.filter((pl.col("pick") != "") | admit)
              .with_columns(pick=pl.col("pick").cast(pl.Int64, strict=False)))
    final = (picked.sort("year")
             .group_by("pid", maintain_order=True).last()
             .with_columns(name_n=pl.col("raw_n").replace(NICKNAME_ALIASES)))

    dh = (pl.read_csv(DATA / "draft_history.csv")
          .filter(pl.col("SEASON").is_between(2011, 2022))
          .with_columns(name_n=pl.col("PLAYER_NAME")
                        .map_elements(norm_name, return_dtype=pl.String)))
    college = dh.filter(pl.col("ORGANIZATION_TYPE") == "College/University")

    by_name = final.join(dh.select("name_n", "PERSON_ID", "SEASON",
                                   "OVERALL_PICK", "ORGANIZATION_TYPE"),
                         on="name_n", how="inner")
    # A name colliding across classes joins twice; keep the row whose final
    # college season matches the draft season. (Torvik finals belonging to
    # 2023+ draftees never reach by_name: their draft record is outside the
    # 2011-2022 window.)
    joined = by_name.filter(pl.col("year") == pl.col("SEASON"))
    return {"final": final, "dh": dh, "college": college,
            "by_name": by_name, "joined": joined}


def build_dataset() -> tuple[pl.DataFrame, dict[str, float]]:
    """Join Torvik final college seasons to draft picks to NBA minutes;
    returns the modeling frame plus the match diagnostics the gates use."""
    m = match_draftees()
    final, college = m["final"], m["college"]
    by_name, joined = m["by_name"], m["joined"]

    year_agree = joined.height / by_name.height
    lottery = college.filter(pl.col("OVERALL_PICK") <= 14)
    lottery_matched = lottery.join(joined.select("PERSON_ID"), on="PERSON_ID",
                                   how="semi").height / lottery.height

    mins = pl.concat([
        pl.read_csv(p).with_columns(
            start_year=pl.lit(int(p.stem.split("_")[1][:4])))
        for p in sorted((DATA / "nba_min").glob("min_*.csv"))])
    first3 = (joined.select("PERSON_ID", "SEASON")
              .join(mins.rename({"PLAYER_ID": "PERSON_ID"}),
                    on="PERSON_ID", how="left")
              .filter((pl.col("start_year") >= pl.col("SEASON"))
                      & (pl.col("start_year") <= pl.col("SEASON") + 2))
              .group_by("PERSON_ID").agg(min3=pl.col("MIN").sum()))

    df = (joined.join(first3, on="PERSON_ID", how="left")
          .with_columns(pl.col("min3").fill_null(0.0))
          .with_columns(
              rotation=(pl.col("min3") >= ROTATION_MIN).cast(pl.Int64),
              log_pick=pl.col("OVERALL_PICK").cast(pl.Float64).log(),
              ht_in=pl.col("ht").map_elements(ht_inches, return_dtype=pl.Float64),
              age=pl.struct("dob", "SEASON").map_elements(
                  lambda s: age_at_draft(s["dob"], s["SEASON"]),
                  return_dtype=pl.Float64),
              class_num=pl.col("yr").replace_strict(CLASS_NUM, default=None)
              .cast(pl.Float64),
              three_a_pg=pl.col("three_a").cast(pl.Float64)
              / pl.col("g").cast(pl.Float64))
          .with_columns(pl.col("obpm", "dbpm", "usg", "ts_per", "orb_per",
                               "drb_per", "ast_per", "to_per", "blk_per",
                               "stl_per", "ftr").cast(pl.Float64))
          .drop_nulls(FEATURES))

    diags = {"year_agree": year_agree, "lottery_matched": lottery_matched,
             "n_final": final.height, "n_model": df.height,
             "base_rate": float(df["rotation"].mean())}
    return df, diags


def fit_and_score(df: pl.DataFrame, features: list[str]) -> dict:
    train = df.filter(pl.col("SEASON").is_in(list(TRAIN_YEARS)))
    test = df.filter(pl.col("SEASON").is_in(list(TEST_YEARS)))
    Xtr = np.column_stack([np.ones(train.height)]
                          + [train[f].to_numpy() for f in features])
    Xte = np.column_stack([np.ones(test.height)]
                          + [test[f].to_numpy() for f in features])
    ytr = train["rotation"].to_numpy().astype(float)
    yte = test["rotation"].to_numpy().astype(float)
    beta = logistic_irls(Xtr, ytr)
    ptr, pte = predict(Xtr, beta), predict(Xte, beta)
    return {"beta": beta, "names": ["intercept"] + features,
            "train_auc": auc(ptr, ytr), "test_auc": auc(pte, yte),
            "train_logloss": log_loss(ptr, ytr),
            "test_logloss": log_loss(pte, yte),
            "pred_test": pte, "y_test": yte, "test": test}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    df, diags = build_dataset()
    print(f"{diags['n_model']} drafted college players modeled "
          f"(base rate {diags['base_rate']:.3f})")
    df.select("player_name", "team", "SEASON", "OVERALL_PICK", "min3",
              "rotation", *FEATURES).write_csv(OUT / "model_frame.csv")

    m0 = fit_and_score(df, ["log_pick"])
    m1 = fit_and_score(df, FEATURES)

    pl.DataFrame({"term": m1["names"], "coef": m1["beta"]}).write_csv(
        OUT / "coefficients_m1.csv")
    pl.DataFrame({"term": m0["names"], "coef": m0["beta"]}).write_csv(
        OUT / "coefficients_m0.csv")

    metrics = pl.DataFrame({
        "model": ["pick_only", "pick_plus_college"],
        "train_auc": [m0["train_auc"], m1["train_auc"]],
        "test_auc": [m0["test_auc"], m1["test_auc"]],
        "train_logloss": [m0["train_logloss"], m1["train_logloss"]],
        "test_logloss": [m0["test_logloss"], m1["test_logloss"]],
    })
    metrics.write_csv(OUT / "metrics.csv")
    print(metrics)

    quintile_calibration(m1["pred_test"], m1["y_test"]).write_csv(
        OUT / "calibration_test.csv")

    surprises = (m1["test"]
                 .with_columns(pred=pl.Series(m1["pred_test"]))
                 .select("player_name", "SEASON", "OVERALL_PICK", "min3",
                         "rotation", "pred")
                 .with_columns(resid=pl.col("rotation") - pl.col("pred")))
    surprises.sort("resid", descending=True).head(10).write_csv(
        OUT / "model_beat_scouts.csv")
    surprises.sort("resid").head(10).write_csv(OUT / "scouts_beat_model.csv")

    checks = pl.DataFrame({
        "check": [
            "inferred draft year agrees with DraftHistory (share of name matches)",
            "lottery picks from colleges matched to Torvik",
            "base rate of rotation outcome in a sane band",
            "pick-only test AUC beats coin flip decisively",
        ],
        "value": [diags["year_agree"], diags["lottery_matched"],
                  diags["base_rate"], m0["test_auc"]],
        "threshold": [0.97, 0.98, 0.25, 0.65],
        "direction": [">=", ">=", ">=", ">="],
    }).with_columns(passed=pl.col("value") >= pl.col("threshold"))
    # base rate also bounded above
    checks = checks.vstack(pl.DataFrame({
        "check": ["base rate bounded above"],
        "value": [diags["base_rate"]], "threshold": [0.60],
        "direction": ["<="], "passed": [diags["base_rate"] <= 0.60]}))
    checks.write_csv(OUT / "validation.csv")
    print(checks)

    import plotly.graph_objects as go
    picks = np.arange(1, 61, dtype=float)
    curve = predict(np.column_stack([np.ones(60), np.log(picks)]), m0["beta"])
    # The curve as a committed table, so the prose's pick-1 value and
    # coin-flip pick are bound to cells rather than recomputed for a figure.
    gap = np.abs(curve - 0.5)
    pl.DataFrame({"pick": picks.astype(int), "p_rotation": curve,
                  "coin_flip": (gap == gap.min()).astype(int)}).write_csv(OUT / "pick_curve.csv")
    f = go.Figure()
    f.add_trace(go.Scatter(x=picks, y=curve, mode="lines",
                           name="P(rotation) from pick alone"))
    # Mark the point the story reads off the curve: where it reaches a coin flip.
    cf = int(picks[gap.argmin()])
    f.add_hline(y=0.5, line=dict(color="#898781", width=1, dash="dash"))
    f.add_annotation(x=cf, y=0.5, text=f"coin flip at pick {cf}", showarrow=False,
                     xanchor="left", yanchor="bottom", xshift=6, yshift=4)
    f.update_layout(title=f"Rotation odds fall steeply from the top pick and reach a "
                    f"coin flip at pick {cf}<br><sup>P(2,000+ NBA minutes in 3 "
                    "seasons) by pick, pick-only model fit on 2011-2018</sup>",
                    xaxis_title="overall pick", yaxis_title="P(rotation)",
                    template="plotly_white")
    f.write_html(FIG / "fig1_pick_curve_py.html", include_plotlyjs="cdn")

    ok = bool(checks["passed"].all())
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
