"""Harvest the draft-model inputs. All cached; re-runs touch nothing fetched.

Three sources:
  1. barttorvik.com getadvstats CSV, 2011-2022 — college advanced stats.
     Free, no key (the cbbdata/toRvik ecosystem's upstream). The endpoint
     returns no header; the 67-column mapping below was pinned empirically
     (Edey 7-4/25.2 ppg/pick 9, Sheppard pick 3, Castle pick 4) and is
     re-verified by a gate in 02_model.py.
  2. nba_api DraftHistory — draft year, pick, and the NBA PERSON_ID that
     lets minutes join by ID instead of by name.
  3. nba_api LeagueDashPlayerStats totals, 2011-12 .. 2024-25 — NBA minutes
     (the outcome). LeagueDash family + the User-Agent fix from
     ../../lineup-valuation-study/python/05_build_stints.py.

Run: python python/01_harvest.py
"""
from __future__ import annotations

import csv
import io
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TORVIK_YEARS = range(2011, 2023)          # draft classes 2011-2022
NBA_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2011, 2025)]

TORVIK_HEADER = [
    "player_name", "team", "conf", "g", "min_per", "o_rtg", "usg", "e_fg",
    "ts_per", "orb_per", "drb_per", "ast_per", "to_per", "ftm", "fta",
    "ft_per", "two_m", "two_a", "two_per", "three_m", "three_a", "three_per",
    "blk_per", "stl_per", "ftr", "yr", "ht", "num", "porpag", "adjoe", "pfr",
    "year", "pid", "type", "rec_rank", "ast_tov", "rim_made", "rim_att",
    "mid_made", "mid_att", "rim_pct", "mid_pct", "dunk_made", "dunk_att",
    "dunk_pct", "pick", "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm",
    "dbpm", "gbpm", "mp", "ogbpm", "dgbpm", "oreb", "dreb", "treb", "ast",
    "stl", "blk", "pts", "role", "x66", "dob",
]


def patch_nba_api_ua() -> None:
    """stats.nba.com hangs UAs that claim a browser the TLS fingerprint
    can't back up (see docs/findings/nba-data-tooling-gotchas.md)."""
    from nba_api.stats.library import http as nba_http
    nba_http.STATS_HEADERS["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")


def harvest_torvik() -> None:
    out = DATA / "torvik_2011_2022.csv"
    if out.exists():
        print(f"already present: {out}")
        return
    rows: list[dict] = []
    for year in TORVIK_YEARS:
        url = f"https://barttorvik.com/getadvstats.php?year={year}&csv=1"
        raw = urllib.request.urlopen(url, timeout=60).read().decode()
        year_rows = [dict(zip(TORVIK_HEADER, r))
                     for r in csv.reader(io.StringIO(raw))]
        assert all(len(r) == len(TORVIK_HEADER) for r in
                   [x for x in csv.reader(io.StringIO(raw))][:5]), year
        rows.extend(year_rows)
        print(f"  torvik {year}: {len(year_rows):,} player-seasons")
        time.sleep(1.0)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TORVIK_HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows):,} rows")


def harvest_draft_history() -> None:
    out = DATA / "draft_history.csv"
    if out.exists():
        print(f"already present: {out}")
        return
    from nba_api.stats.endpoints import drafthistory
    df = drafthistory.DraftHistory(timeout=60).get_data_frames()[0]
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df):,} picks")


def harvest_nba_minutes() -> None:
    from nba_api.stats.endpoints import leaguedashplayerstats
    mins_dir = DATA / "nba_min"
    mins_dir.mkdir(exist_ok=True)
    for season in NBA_SEASONS:
        out = mins_dir / f"min_{season}.csv"
        if out.exists():
            continue
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="Totals",
            timeout=60).get_data_frames()[0]
        df[["PLAYER_ID", "PLAYER_NAME", "GP", "MIN"]].to_csv(out, index=False)
        print(f"  minutes {season}: {len(df):,} players")
        time.sleep(1.0)


def main() -> int:
    DATA.mkdir(exist_ok=True)
    patch_nba_api_ua()
    harvest_torvik()
    harvest_draft_history()
    harvest_nba_minutes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
