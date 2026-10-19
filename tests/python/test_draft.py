"""Unit tests for draft-study/python/02_model.py pure functions."""
from __future__ import annotations

import numpy as np
import pytest


def test_norm_name(draft):
    assert draft.norm_name("Gary Trent Jr.") == "gary trent"
    assert draft.norm_name("Nah'Shon Hyland") == "nahshon hyland"
    assert draft.norm_name("Sviatoslav Mykhailiuk") == "sviatoslav mykhailiuk"
    assert draft.norm_name("Wendell Carter III") == "wendell carter"
    assert draft.norm_name("Dario Šarić") == "dario saric"


def test_ht_inches(draft):
    assert draft.ht_inches("6-10") == 82.0
    assert draft.ht_inches("7-4") == 88.0
    assert draft.ht_inches("") is None
    assert draft.ht_inches("tall") is None


def test_age_at_draft(draft):
    # Born June 25 of year Y, drafted year Y+19 -> exactly 19.
    assert draft.age_at_draft("2000-06-25", 2019) == pytest.approx(19.0)
    assert draft.age_at_draft("", 2019) is None


def test_auc_known_values(draft):
    y = np.array([0.0, 0, 1, 1])
    assert draft.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert draft.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    # All-tied predictions -> 0.5 by the average-rank convention.
    assert draft.auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == pytest.approx(0.5)


def test_log_loss_matches_formula(draft):
    # Both cases contribute -log(0.9): a hit at 0.9 and a miss at 0.1.
    pred = np.array([0.9, 0.1])
    y = np.array([1.0, 0.0])
    assert draft.log_loss(pred, y) == pytest.approx(-np.log(0.9))


def test_join_gate_passes_on_committed_tables(draft):
    gate = draft_gate()
    assert gate.audit(draft) == []


def test_join_gate_fails_when_allowlist_row_removed(draft):
    import polars as pl
    gate = draft_gate()
    allow = pl.read_csv(draft.ALLOWLIST_FILE, infer_schema_length=0)
    problems = gate.audit(draft, allow.filter(pl.col("player_name") != "Usman Garuba"))
    assert any("Usman Garuba" in p and "UNMATCHED" in p for p in problems)


def test_join_gate_flags_stale_allowlist_row(draft):
    import polars as pl
    gate = draft_gate()
    allow = pl.read_csv(draft.ALLOWLIST_FILE, infer_schema_length=0)
    extra = pl.DataFrame({"person_id": ["202717"], "player_name": ["Trey Thompkins"],
                          "season": ["2011"], "overall_pick": ["37"],
                          "kind": ["unmatched"], "reason": ["x"]})
    problems = gate.audit(draft, pl.concat([allow, extra]))
    assert any("STALE" in p and "202717" in p for p in problems)


def test_recovered_draftees_are_in_the_join(draft):
    names = set(draft.match_draftees()["joined"]["player_name"])
    assert {"Howard Thompkins III", "Balsa Koprivica"} <= names


def draft_gate():
    from conftest import load_script
    return load_script("python/05_join_gate.py", "draft_join_gate")
