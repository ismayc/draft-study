"""Explainer diagrams for the story page, drawn with the family's famdiagram
(../basketball-analysis-tools/famdiagram.py) and saved to figures/<ref>.svg
(plus a <ref>_narrow.svg phone layout). report_builder.py inlines them.

Every number drawn is read from a committed table: output/model_frame.csv
(the per-player modeling frame python/02_model.py writes) and
output/metrics.csv (reconciled with R by python/03_reconcile.py). The split
years, the rotation threshold, and the feature list are read from
python/02_model.py itself, so the diagrams cannot drift from the model.

Diagrams:
  dg1_era_holdout  every modeled draftee as a dot in his draft class, colored
                   by whether he reached a rotation, with the classes the
                   model is fit on and the classes it is scored on
                   (output/model_frame.csv)
  dg2_pair_auc     what a test AUC means: every held-out pair of one player
                   who reached a rotation and one who did not, sorted by
                   whether the earlier pick was the one who made it. The
                   pair count reproduces the pick-only test AUC exactly, and
                   the script refuses to draw if it does not
                   (output/model_frame.csv, output/metrics.csv)
  dg3_in_sample    the in-sample gain from the college features against the
                   out-of-era result, for both models (output/metrics.csv)

Run:     python python/06_diagrams.py
--check: rebuild every SVG in memory and fail if a committed one differs.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT.parent / "basketball-analysis-tools"))
import famdiagram as fd  # noqa: E402

MINUS = "−"
# Plain-language names for the college features in python/02_model.py.
FEATURE_LABELS = {
    "age": "age at draft", "ht_in": "height", "obpm": "offensive BPM",
    "dbpm": "defensive BPM", "usg": "usage rate", "ts_per": "true shooting",
    "orb_per": "offensive rebounding", "drb_per": "defensive rebounding",
    "ast_per": "assist rate", "to_per": "turnover rate", "blk_per": "block rate",
    "stl_per": "steal rate", "three_a_pg": "threes attempted",
    "ftr": "free-throw rate", "class_num": "college class",
}


def model_module():
    spec = importlib.util.spec_from_file_location(
        "draft_model_for_diagrams", ROOT / "python" / "02_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = model_module()


def signed(v: float, d: int = 3) -> str:
    s = f"{abs(v):.{d}f}"
    if float(s) == 0:
        return s
    return ("+" if v > 0 else MINUS) + s


def years(r: range) -> str:
    return f"{r.start}-{r.stop - 1}"


def wrap(s: str, width: float, size: float) -> list[str]:
    """Greedy word wrap on an estimated glyph width."""
    per = size * 0.55
    out, cur = [], ""
    for w in s.split():
        t = f"{cur} {w}".strip()
        if len(t) * per > width and cur:
            out.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        out.append(cur)
    return out


def para(d: fd.Diagram, x, y, s, width, size, lh=None, **kw) -> float:
    """Wrapped text; returns the baseline of the last line."""
    lh = lh or size * 1.35
    lines = wrap(s, width, size)
    for i, line in enumerate(lines):
        d.text(x, y + i * lh, line, size=size, **kw)
    return y + (len(lines) - 1) * lh


def pair_counts(frame: pl.DataFrame) -> dict:
    """Held-out (rotation, no rotation) pairs by which one was picked first.
    The pick-only model ranks players by pick, so (earlier + ties / 2) / pairs
    is its test AUC."""
    test = frame.filter(pl.col("SEASON").is_in(list(M.TEST_YEARS)))
    made = test.filter(pl.col("rotation") == 1)["OVERALL_PICK"].to_list()
    miss = test.filter(pl.col("rotation") == 0)["OVERALL_PICK"].to_list()
    right = sum(a < b for a in made for b in miss)
    tied = sum(a == b for a in made for b in miss)
    pairs = len(made) * len(miss)
    return {"made": len(made), "miss": len(miss), "pairs": pairs, "right": right,
            "tied": tied, "wrong": pairs - right - tied,
            "auc": (right + tied / 2) / pairs, "test": test}


# ---------------------------------------------------------------- dg1
def era_holdout(narrow: bool) -> fd.Diagram:
    f = pl.read_csv(OUT / "model_frame.csv")
    classes = sorted(f["SEASON"].unique().to_list())
    W = 360 if narrow else 760
    fs, small, head = (13, 12, 17) if narrow else (16, 14, 20)
    pad = 12
    d = fd.Diagram(W, 10, "Fit on the earlier draft classes, scored on the later ones")
    y = 28
    y = para(d, pad, y, "Every modeled draftee, one dot, in his draft class",
             W - 2 * pad, head, display=True, weight="600")
    y += 24
    d.circle(pad + 6, y - 4, 5, fill="nba")
    lab = f"{M.ROTATION_MIN:,.0f}+ NBA minutes in his first three seasons"
    d.text(pad + 16, y, lab, size=small, fill="ink2")
    y += 20
    d.circle(pad + 6, y - 4, 5, fill="ink3", opacity=0.5)
    d.text(pad + 16, y, "fewer", size=small, fill="ink2")
    y += 22
    label_w = 44 if narrow else 52
    side_w = 0 if narrow else 176
    dots_w = W - 2 * pad - label_w - side_w
    widest = int(f.group_by("SEASON").len()["len"].max())
    sp = dots_w / widest
    rad = min(sp * 0.4, 3.6)
    rh = 15 if narrow else 17
    blocks = [("fit", M.TRAIN_YEARS), ("score", M.TEST_YEARS)]
    for bi, (kind, yrs) in enumerate(blocks):
        cls = [c for c in classes if c in yrs]
        sub = f.filter(pl.col("SEASON").is_in(cls))
        top = y
        if kind == "score":
            d.rect(pad - 4, top - 6, W - 2 * pad + 8, len(cls) * rh + 12,
                   fill="surface2", stroke="rule", rx=6)
        for k, c in enumerate(cls):
            row = (f.filter(pl.col("SEASON") == c)
                   .sort(["rotation", "OVERALL_PICK"], descending=[True, False]))
            ry = top + k * rh + rh / 2
            d.text(pad + label_w - 8, ry + small * 0.35, str(c), size=small,
                   fill="ink2", anchor="end")
            for j, made in enumerate(row["rotation"].to_list()):
                cx = pad + label_w + sp * j + sp / 2
                if made:
                    d.circle(cx, ry, rad, fill="nba")
                else:
                    d.circle(cx, ry, rad, fill="ink3", opacity=0.5)
        bottom = top + len(cls) * rh
        n = sub.height
        if kind == "fit":
            t1 = f"Fit on the {years(yrs)} classes"
            t2 = f"{n} players: the model learns here"
        else:
            t1 = f"Scored on {years(yrs)}"
            t2 = f"{n} players the fit never saw"
        if narrow:
            yy = bottom + 22
            d.text(pad, yy, t1, size=fs, fill="ink", weight="700")
            d.text(pad, yy + fs * 1.35, t2, size=small, fill="ink2")
            y = yy + fs * 1.35 + 26
        else:
            sx = W - pad - side_w + 18
            d.line(sx - 8, top + 2, sx - 8, bottom - 2, stroke="ink3", sw=1.5)
            mid = (top + bottom) / 2
            n1 = len(wrap(t1, side_w - 22, small))
            yy = para(d, sx, mid - 4 - (n1 - 1) * small * 0.7, t1, side_w - 22, small,
                      fill="ink", weight="700")
            para(d, sx, yy + small * 1.45, t2, side_w - 22, small, fill="ink2")
            y = bottom + 24
    y += 6 if narrow else 14
    y = para(d, pad, y, f"A random split would scatter test players across all "
             f"{len(classes)} classes, so the model would be scored on the same eras it was fit on. "
             "Holding out whole classes asks the question a front office faces: "
             "will a model fit on past drafts rank the next ones?",
             W - 2 * pad, fs if not narrow else small, fill="ink2")
    d.h = y + 16
    return d


# ---------------------------------------------------------------- dg2
def pair_auc(narrow: bool) -> fd.Diagram:
    f = pl.read_csv(OUT / "model_frame.csv")
    met = pl.read_csv(OUT / "metrics.csv")
    pub = float(met.filter(pl.col("model") == "pick_only")["test_auc"][0])
    pc = pair_counts(f)
    if abs(pc["auc"] - pub) > 1e-12:
        raise SystemExit(f"dg2: pair AUC {pc['auc']} != published {pub}")
    test = pc["test"]
    made = test.filter(pl.col("rotation") == 1).sort(["OVERALL_PICK", "SEASON"])
    miss = test.filter(pl.col("rotation") == 0).sort(["OVERALL_PICK", "SEASON"])
    ex_right = (made.row(0, named=True), miss.row(miss.height - 1, named=True))
    ex_wrong = (miss.row(0, named=True), made.row(made.height - 1, named=True))

    W = 360 if narrow else 760
    fs, small, head = (13, 12, 17) if narrow else (16, 14, 20)
    pad = 12
    d = fd.Diagram(W, 10, "What a test AUC means, counted pair by pair")
    y = 28
    y = para(d, pad, y, f"Pair every held-out player who reached a rotation ({pc['made']}) "
             f"with every one who did not ({pc['miss']})", W - 2 * pad, head,
             display=True, weight="600")
    y += 24
    y = para(d, pad, y, f"{pc['pairs']:,} pairs. For each, ask: was the player who made "
             "it picked earlier?", W - 2 * pad, fs, fill="ink2")
    y += 22
    # example cards
    cw = W - 2 * pad if narrow else (W - 3 * pad) / 2
    ch = 108
    for i, (lab, col, (a, b), verdict) in enumerate((
            ("Earlier pick made it", "nba", ex_right, "counts as right"),
            ("Later pick made it", "c8", ex_wrong, "counts as wrong"))):
        cx = pad if narrow else pad + i * (cw + pad)
        cy = y + (i * (ch + 10) if narrow else 0)
        d.rect(cx, cy, cw, ch, fill="surface2", stroke="rule", rx=8)
        d.text(cx + 12, cy + 24, f"For example: {lab.lower()}", size=small,
               fill=col, weight="700")
        for k, p in enumerate((a, b)):
            ok = "made a rotation" if p["rotation"] else "did not"
            d.text(cx + 12, cy + 48 + k * (small * 1.45),
                   f"#{p['OVERALL_PICK']} {p['player_name']} ({p['SEASON']}): {ok}",
                   size=small, fill="ink")
        d.text(cx + 12, cy + ch - 12, verdict, size=small, fill="ink3", italic=True)
    y += (2 * ch + 10 if narrow else ch) + 36
    # the split bar
    d.text(pad, y, "All held-out pairs", size=fs, fill="ink", weight="700")
    y += 12
    bw = W - 2 * pad
    bh = 30
    segs = [("right", pc["right"], "nba", 1.0), ("tied", pc["tied"], "ink3", 0.5),
            ("wrong", pc["wrong"], "c8", 0.85)]
    x = pad
    for _, v, col, op in segs:
        w = bw * v / pc["pairs"]
        d.rect(x, y, w, bh, fill=col, opacity=op)
        x += w
    y += bh + 22
    labs = [(f"earlier pick made it: {pc['right']:,}", "nba"),
            (f"same pick slot, different years: {pc['tied']}", "ink2"),
            (f"later pick made it: {pc['wrong']:,}", "c8")]
    if narrow:
        for k, (t, col) in enumerate(labs):
            d.text(pad, y + k * 18, t, size=small, fill=col, weight="600")
        y += 2 * 18 + 30
    else:
        d.text(pad, y, labs[0][0], size=small, fill=labs[0][1], weight="600")
        d.text(W - pad, y, labs[2][0], size=small, fill=labs[2][1], weight="600",
               anchor="end")
        d.text(pad, y + 20, labs[1][0], size=small, fill=labs[1][1])
        y += 20 + 34
    d.text(pad, y, "AUC = (right + half the ties) / all pairs", size=fs, fill="ink2")
    y += fs * 1.6 + 6
    d.text(pad, y, f"({pc['right']:,} + {pc['tied']} / 2) / {pc['pairs']:,} = "
           f"{pc['auc']:.3f}", size=fs + 5, fill="nba", weight="700")
    y = para(d, pad, y + 26, "Ranking players at random would get half the pairs right.",
             W - 2 * pad, small, fill="ink3", italic=True)
    d.h = y + 16
    return d


# ---------------------------------------------------------------- dg3
def in_sample(narrow: bool) -> fd.Diagram:
    met = pl.read_csv(OUT / "metrics.csv")
    rows = {r["model"]: r for r in met.iter_rows(named=True)}
    p0, p1 = rows["pick_only"], rows["pick_plus_college"]
    college = [FEATURE_LABELS[x] for x in M.FEATURES if x != "log_pick"]
    vals = [p0["train_auc"], p0["test_auc"], p1["train_auc"], p1["test_auc"]]
    lo, hi = min(vals) - 0.006, max(vals) + 0.004

    W = 360 if narrow else 760
    fs, small, head = (13, 12, 17) if narrow else (16, 14, 20)
    pad = 12
    d = fd.Diagram(W, 10, "The college features help in sample and add nothing out of era")
    y = 28
    y = para(d, pad, y, "Two models, scored where they were fit and on the held-out classes",
             W - 2 * pad, head, display=True, weight="600")
    y += 30
    gw = (W - 2 * pad) if narrow else 470
    x0 = pad + (112 if narrow else 120)
    x1 = pad + gw - (64 if narrow else 120)
    ph = 210
    top = y + 50

    def Y(v):
        return top + ph * (hi - v) / (hi - lo)

    d.text(x0, y, "In sample", size=fs, fill="ink", anchor="middle", weight="700")
    d.text(x0, y + small * 1.3, years(M.TRAIN_YEARS), size=small, fill="ink3",
           anchor="middle")
    d.text(x1, y, "Out of era", size=fs, fill="ink", anchor="middle", weight="700")
    d.text(x1, y + small * 1.3, years(M.TEST_YEARS), size=small, fill="ink3",
           anchor="middle")
    for x in (x0, x1):
        d.line(x, top, x, top + ph, stroke="rule", sw=1.5)
    for r, col, name in ((p0, "ink2", "pick only"), (p1, "nba", "pick + college")):
        a, b = Y(r["train_auc"]), Y(r["test_auc"])
        d.line(x0, a, x1, b, stroke=col, sw=3)
        d.circle(x0, a, 5, fill=col)
        d.circle(x1, b, 5, fill=col)
        d.text(x0 - 10, a + 5, f"{r['train_auc']:.3f}", size=fs, fill=col, anchor="end",
               weight="700")
        if r is p0:
            d.text(x0 - 10, a + 5 + fs * 1.3, name, size=small, fill=col, anchor="end")
        else:
            d.text(x0 - 10, a + 5 - fs * 1.3, name, size=small, fill=col, anchor="end")
    # the two test values nearly coincide: label them once, side by side
    tb = Y(p0["test_auc"])
    d.text(x1 + 10, tb - 2, f"{p0['test_auc']:.3f}", size=fs, fill="ink2", weight="700")
    d.text(x1 + 10, tb + fs * 1.2, f"{p1['test_auc']:.3f}", size=fs, fill="nba",
           weight="700")
    yb = top + ph + 30

    gain_in = p1["train_auc"] - p0["train_auc"]
    gain_out = p1["test_auc"] - p0["test_auc"]
    if narrow:
        bx, by0, bwid = pad, yb, W - 2 * pad
    else:
        bx, by0, bwid = pad + gw + 20, y - 10, W - (pad + gw + 20) - pad
    tx, tw = bx + 14, bwid - 28
    box_at = len(d.parts)
    yy = by0 + 30
    d.text(tx, yy, "What the college stats add", size=fs, fill="ink", weight="700")
    d.text(tx, yy + 34, signed(gain_in), size=30, display=True, weight="700", fill="ink2")
    d.text(tx + 92, yy + 34, "in sample", size=small, fill="ink2")
    d.text(tx, yy + 72, signed(gain_out), size=30, display=True, weight="700", fill="nba")
    d.text(tx + 92, yy + 72, "out of era", size=small, fill="ink2")
    yy = para(d, tx, yy + 104, f"The {len(college)} college features:", tw, small,
              fill="ink2", weight="600")
    yy = para(d, tx, yy + small * 1.4, ", ".join(college), tw, small, fill="ink3")
    bh = yy + 18 - by0
    d.rect(bx, by0, bwid, bh, fill="surface2", stroke="rule", rx=8)
    d.parts.insert(box_at, d.parts.pop())
    d.h = max(yb, by0 + bh) + 16
    return d


DIAGRAMS = {"dg1_era_holdout": era_holdout, "dg2_pair_auc": pair_auc,
            "dg3_in_sample": in_sample}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        fails = []
        with tempfile.TemporaryDirectory() as tmp:
            for ref, fn in DIAGRAMS.items():
                for p in fd.save(Path(tmp), ref, fn(False), fn(True)):
                    old = FIG / p.name
                    if not old.exists() or old.read_text() != p.read_text():
                        fails.append(p.name)
        for f in fails:
            print("FAIL", f, "differs from a fresh build")
        print("DIAGRAMS", "FAILED" if fails else
              f"PASSED: {len(DIAGRAMS)} diagrams rebuild identically")
        return 1 if fails else 0
    for ref, fn in DIAGRAMS.items():
        for p in fd.save(FIG, ref, fn(False), fn(True)):
            print("wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
