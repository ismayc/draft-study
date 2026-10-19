"""Gate: every 2011-2022 college draftee joins to a Torvik final season, or
is listed with a reason in data/draft_join_allowlist.csv.

02_model.py joins Torvik to DraftHistory by normalized name with an inner
join, so an unmatched draftee drops out silently. This gate makes the drop
loud. It fails (exit 1, every offender printed) when:

  1. a college draftee is unmatched and not allowlisted (kind "unmatched");
  2. an allowlisted "unmatched" draftee now matches (stale entry);
  3. a matched player's Torvik pick differs from DraftHistory's and the
     player is not allowlisted (kind "pick_mismatch"), or an allowlisted
     mismatch has stopped disagreeing;
  4. an alias or allowlist row has no reason.

Run: python python/05_join_gate.py   (reads only committed repo data)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def _model():
    spec = importlib.util.spec_from_file_location(
        "draft_model_gate", ROOT / "python" / "02_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["draft_model_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def audit(model, allowlist: pl.DataFrame | None = None) -> list[str]:
    """Return one problem line per violation (empty list means pass)."""
    if allowlist is None:
        allowlist = pl.read_csv(model.ALLOWLIST_FILE, infer_schema_length=0)
    allow_unmatched = set(allowlist.filter(pl.col("kind") == "unmatched")
                          ["person_id"].cast(pl.Int64))
    allow_pick = set(allowlist.filter(pl.col("kind") == "pick_mismatch")
                     ["person_id"].cast(pl.Int64))
    m = model.match_draftees()
    college, joined = m["college"], m["joined"]
    problems: list[str] = []

    matched_ids = set(joined["PERSON_ID"])
    unmatched = college.filter(~pl.col("PERSON_ID").is_in(list(matched_ids)))
    for r in unmatched.sort("SEASON", "OVERALL_PICK").iter_rows(named=True):
        if r["PERSON_ID"] not in allow_unmatched:
            problems.append(
                f"UNMATCHED college draftee not allowlisted: {r['PLAYER_NAME']}"
                f" (id {r['PERSON_ID']}, {r['SEASON']} pick {r['OVERALL_PICK']},"
                f" {r['ORGANIZATION']})")
    for pid in sorted(allow_unmatched & matched_ids):
        problems.append(f"STALE allowlist row: id {pid} now matches; remove it")
    for pid in sorted(allow_unmatched - set(college["PERSON_ID"])):
        problems.append(f"STALE allowlist row: id {pid} is not a 2011-2022"
                        " college draftee")

    disagree = joined.filter(pl.col("pick").is_not_null()
                             & (pl.col("pick") != pl.col("OVERALL_PICK")))
    bad_pick_ids = set(disagree["PERSON_ID"])
    for r in disagree.sort("SEASON", "OVERALL_PICK").iter_rows(named=True):
        if r["PERSON_ID"] not in allow_pick:
            problems.append(
                f"PICK MISMATCH not allowlisted: {r['player_name']} (id "
                f"{r['PERSON_ID']}, {r['SEASON']}) Torvik pick {r['pick']} vs "
                f"DraftHistory {r['OVERALL_PICK']}")
    for pid in sorted(allow_pick - bad_pick_ids):
        problems.append(f"STALE allowlist row: id {pid} no longer has a pick"
                        " mismatch")

    aliases = pl.read_csv(model.ALIAS_FILE, infer_schema_length=0)
    for name in aliases.filter(pl.col("reason").fill_null("").str.strip_chars()
                               == "")["torvik_name_n"]:
        problems.append(f"alias without a reason: {name}")
    for pid in allowlist.filter(pl.col("reason").fill_null("")
                                .str.strip_chars() == "")["person_id"]:
        problems.append(f"allowlist row without a reason: id {pid}")
    return problems


def main() -> int:
    model = _model()
    problems = audit(model)
    m = model.match_draftees()
    n_col = m["college"].height
    n_join = m["college"].join(m["joined"].select("PERSON_ID"), on="PERSON_ID",
                               how="semi").height
    allow = pl.read_csv(model.ALLOWLIST_FILE, infer_schema_length=0)
    n_allow = allow.filter(pl.col("kind") == "unmatched").height
    print(f"{n_col} college draftees 2011-2022: {n_join} matched, "
          f"{n_allow} allowlisted with a reason, "
          f"{n_col - n_join - n_allow} unexplained")
    for p in problems:
        print("  FAIL", p)
    print("JOIN GATE PASSED" if not problems else "JOIN GATE FAILED")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
