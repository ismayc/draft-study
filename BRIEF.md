# Draft model — the same findings for three audiences

## For the front office

We asked whether a drafted player's college box-score numbers tell you
anything his draft slot doesn't. On twelve draft classes, the answer for
this outcome is no: a model that knows only where a player was picked
predicts "rotation player within three years" exactly as well as one that
also reads his college stat line. The pick already contains the stat line —
scouts price the public numbers in. Two practical reads: first, the value
curve by pick (near-certain at the top, a coin flip by the late first) is a
defensible baseline for trade and pick-swap valuations; second, if a draft
model is going to beat the board, it needs inputs the board hasn't seen —
tracking data, medical, intel — not another pass over the box score.

## For analytics peers

n=567 drafted college players, classes 2011-2022 (Torvik advanced stats;
outcome ≥2,000 NBA minutes in three seasons via DraftHistory PERSON_ID join
to LeagueDash totals — the only name matching is Torvik↔DraftHistory, gated
at 98%+ with a hand-audited alias table). Logistic M0 (log pick) vs M1
(+15 college features), same hand-rolled IRLS as the shot-quality study,
independently implemented in R and Python, reconciling at ~1e-13. Era
holdout (train 2011-2018, test 2019-2022): M1 gains +0.017 AUC in-sample
and +0.000 out-of-sample with worse log-loss — clean overfit, reported as
the finding rather than hidden behind the train table. Random CV would have
flattered M1; the time split is the honest design. Limitations stated:
blunt outcome, college-only frame, n too small to close the question.

## For the executive summary (three bullets)

- Tested whether public college stats add draft insight beyond the pick
  itself, on twelve draft classes with a held-out four-class test.
- They don't — the scouting consensus already prices the box score in; the
  in-sample "improvement" vanishes on drafts the model never saw.
- Byproduct with standalone value: a calibrated probability-by-pick curve
  (rotation player: near-certain at #1, coin flip by the late first) for
  valuing picks in trades.
