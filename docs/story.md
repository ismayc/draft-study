# The scouts already priced it in: a draft model whose finding is "nothing"

Every June, analysts publish college-stats models that claim to out-draft
NBA front offices. This study asks the question those models usually skip:
given where a player was actually picked, does his college production tell
you *anything more* about whether he will make it? The draft pick is not a
control variable to brush aside. It **is** the scouting industry's
consensus forecast, assembled from workouts, interviews, medicals, and
thousands of scout-hours. Beating it is the bar.

<div class="tiles">
<div class="tile"><div class="label">What college stats add</div>
<div class="value">0.000<small> AUC</small></div>
<div class="sub">Fifteen box-score features move the out-of-sample test AUC not at all.</div></div>
<div class="tile"><div class="label">The pick alone</div>
<div class="value">0.829<small> AUC</small></div>
<div class="sub">Out-of-era prediction of rotation outcomes from draft position only.</div></div>
<div class="tile"><div class="label">In-sample flattery</div>
<div class="value">0.844 &rarr; 0.862</div>
<div class="sub">The gain a random split would have reported as skill.</div></div>
<div class="tile"><div class="label">Sample</div>
<div class="value">569</div>
<div class="sub">Drafted college players, 2011&ndash;2022, every historical mismatch in a committed alias table or allowlist.</div></div>
</div>

<div class="eyebrow beat">The design</div>

## An era holdout keeps the model from peeking

The design is built to make the answer trustworthy rather than flattering.
569 drafted college players (2011–2022) from Barttorvik's advanced-stats
archive, joined to NBA outcomes by player ID, with the study's one name
join gated at 98% and every historical mismatch in a committed alias table
or allowlist. The outcome is concrete: a real rotation (2,000+ NBA minutes
within three seasons). And the holdout is **by era** (train on the
2011–2018 classes, test on 2019–2022), because a random split would let the
model peek at the era it claims to predict, which is exactly how draft
models flatter themselves.

{{svg:dg1_era_holdout|The era holdout. Each dot is one drafted college player in his draft class, blue if he reached a rotation. The model is fit only on the upper block of classes and scored only on the shaded block, which it never sees while fitting, so its score measures how well it ranks drafts it has not seen.}}

**Corrected October 19, 2026.** The name join used to drop college
draftees without a word. An audit found the dropped players; two whose
records were complete are back in the sample, the rest are listed with a
reason, and a gate now fails on any other. Every figure on this page moved
by a few thousandths at most, and the conclusion did not change.

<div class="eyebrow beat">The result</div>

## The college numbers add 0.000

The result: a model using only the pick predicts rotation outcomes at
0.829 AUC out of sample. AUC is easiest to read as a count over pairs of
players, drawn below on the held-out classes.

{{svg:dg2_pair_auc|What the test AUC counts. Every held-out player who reached a rotation is paired with every one who did not, and a pair counts as right when the earlier pick is the one who made it. The pick-only model ranks players by pick, so this count reproduces its test AUC exactly, and the scouts' order gets most pairs right.}}

Add fifteen college box-score features (scoring,
rebounding, usage, stocks, age, height, the works) and the test AUC moves
to 0.829. **The college numbers add 0.000.** In-sample they help (0.844 →
0.862), which is precisely the trap: a random-split evaluation would have
reported that in-sample flattery as skill. Out of era, the scouts' consensus
already contains everything the box score knows.

{{svg:dg3_in_sample|The in-sample trap. Scored on the classes it was fit on, the model with college features looks better than the pick alone; scored on the held-out classes, the two land in the same place. The gain on the left is what a random-split evaluation would have reported as skill.}}

<div class="eyebrow beat">The byproduct</div>

## The negative result pays for a pick-value curve

A negative result is only useful if it is stated as one, so this study
promotes it to the headline instead of burying it: public college
production, on its own, is not an edge over the pick. The honest
byproduct, though, is genuinely useful, because modeling P(rotation) as a
function of the pick produces the empirical pick-value curve every trade
conversation needs.

{{fig:fig1_pick_curve_py|P(NBA rotation player within three seasons) by draft pick: the study's one essential figure. It converts a negative modeling result into a usable asset-valuation tool: the curve is steep at the top, crosses a coin flip around pick 28, and flattens into the second round. When a trade offer bundles picks, this curve is the exchange rate; hover any pick for its modeled probability.}}

<div class="eyebrow beat">Reading the curve</div>

## Late-first-round certainty is a coin toss

Two readings of the curve do real work in front-office arguments. First,
its steepness at the top says the difference between pick 3 and pick 12 is
enormous. Consensus is *informative*, which is exactly why beating it is
hard. Second, the coin-flip crossing near the back of the first round says
late-first-round certainty is an illusion. A team trading up at pick 27
"because our model loves him" is paying a premium for a coin toss, unless
its information is genuinely private.

<div class="eyebrow beat">The lesson</div>

## An edge has to come from information the consensus doesn't hold

That is the study's actual thesis, and it is a decision-making principle
rather than a model: **an edge has to come from information the consensus
does not already hold** (medicals, workouts, intel, or data nobody else
has). A model built from public inputs and evaluated the honest way shows
you the ceiling of public information, and knowing that ceiling is worth
more to a front office than another flattering backtest. The same
discipline as the rest of the family applies (dual R/Python
implementation reconciled to ~1e-13, with the era holdout wired into the
gates), because a negative result you cannot trust is worth nothing at
all.
