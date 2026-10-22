## Decision after evaluation 30
**Budget:** 30 used / 275 total; 245 remaining.
**Evidence:** The 30 MaxMin-diverse candidates gave y in [-23.62, 2.93],
mean -4.83, median -4.23, stdev 5.63 — a wide, clearly non-flat distribution.
Best so far: idx 409 (`OC[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)CO`, a hexitol
sugar alcohol), y=-23.62. Other low-y hits: idx 159 (a fluorinated uracil,
-15.46), idx 475 (caffeine, -12.64), idx 84 (nitroaniline, -9.82), idx 437
(hydrazine, -9.3), idx 212 (DMF, -7.81). These low-y molecules are chemically
diverse in scaffold but share small size and high polarity/H-bonding
capacity. I computed RDKit descriptors (MolWt, TPSA, Crippen logP, HBD, HBA,
rotatable bonds, aromatic ring count) for all 30 observed points and
correlated each with y: TPSA r=-0.90, HBD r=-0.78, HBA r=-0.69, logP r=+0.55,
MW r=-0.18. This is a strong, consistent signal (more polar/more H-bonding/
less lipophilic -> lower y) rather than noise — it holds across chemically
unrelated scaffolds, so it looks like a real descriptor-level relationship
rather than an artifact of one cluster.
I fit a ridge regression (alpha=2.0 on standardized features, closed-form,
numpy) on these 7 descriptors against y and checked it with leave-one-out
cross-validation (honest, out-of-sample per point): LOO RMSE=2.53, LOO
R2=0.79 (y stdev=5.54). This is a meaningfully predictive surrogate, not
just an in-sample fit (in-sample R2=0.90 was close to the LOO figure, so no
sign of catastrophic overfitting with only 7 features and 30 points).
**Decision:** Switch from diversity sampling to **surrogate-guided
exploitation**: rank all 479 remaining unevaluated candidates by predicted y
from the ridge model (fit on all 30 current observations), and evaluate the
top-ranked (most negative predicted y) candidates next.
**Hyperparameters:** Ridge alpha=2.0 (intercept unregularized), features
standardized (z-score) using training-set mean/std, closed-form solve via
numpy.linalg.solve — a small, deliberately simple/low-variance model given
n=30. Not inherited from any library default; chosen directly. No other
tuning performed (alpha was picked as a reasonable, mildly-regularizing
value for p=7, n=30; not swept, since LOO already confirmed it generalizes
acceptably).
**Reason:** With 245 evaluations remaining and a cheap, verified-predictive
(LOO R2=0.79) descriptor-based model available, pure exploitation of the
surrogate's ranking is the most promising use of the next block of budget.
A moderate batch (not the full remaining budget) is chosen so the surrogate
can be refit and re-validated with more data before committing further,
since LOO R2=0.79 still leaves real residual uncertainty and the model could
be systematically wrong in some region of descriptor space not yet sampled.
**Next step limit:** 40 additional objective calls, at most 245; pause at
cumulative evaluation 70.
**Why this length:** Large enough to meaningfully test whether the
surrogate's top-ranked candidates actually achieve lower y than the
diversity-sampling batch did (a real test of the TPSA/HBD/logP relationship
outside the training sample), and to give enough new data to meaningfully
refit/re-validate the surrogate next time. Small enough (40/245 ≈ 16%) that
if the surrogate turns out to be miscalibrated or the top predictions
cluster in an already-exhausted region, most of the budget is still
available to adapt.
**Review:** At evaluation 70, check: (1) whether the realized y values for
this batch are in fact lower than the batch-30 distribution (confirming the
surrogate adds value over random/diverse picking), (2) updated LOO R2 on the
combined 70-point set, (3) whether the new best-so-far improves on -23.62,
and (4) whether the top-ranked candidates were concentrated in one narrow
structural family (which would suggest needing to diversify again to avoid
redundant sampling). No preselected next method.

## Decision after evaluation 70
**Budget:** 70 used / 275 total; 205 remaining.
**Evidence:** The 40 surrogate-ranked candidates (`ridge_surrogate_exploit_v1`)
scored mean y=-10.40, stdev=4.45, min=-20.52, max=-2.45 — substantially
better than the random/diverse batch-1 mean of -4.83, and every single value
in this batch was negative (vs. several positive values in batch 1). This
confirms the TPSA/HBD/logP-driven ridge surrogate adds real value over
diversity sampling. New best-so-far is still idx 409 (-23.62, unchanged);
second-best is now idx 151 (-20.52, another hexitol sugar alcohol). The top
of the combined ranking is dominated by two structural families: (a) uracil
with a halogen at the 5-position — I had, by chance/diversity-picking,
already tested the parent (idx 159, F) and the surrogate then surfaced Cl
(326), Br (106), I (379) analogs, all strongly negative (-16.9 to -18.7),
plus a close relative, triuret/isocyanuric-acid-like idx 503 (-18.06); and
(b) polyol sugars (two hexitols so far, -23.62 and -20.52).
I refit the ridge surrogate on all 70 points (same alpha=2.0, same
procedure): LOO R2 improved from 0.79 to 0.84, LOO RMSE improved from 2.53 to
2.26 — the model is getting more reliable with more data, not overfitting
worse. However, its new top-50 predictions for the remaining 439 unused
candidates are much less extreme (best predicted ~-9.7) than the actual
observed extremes (-23.62), which is expected: the handful of most-extreme
polar/small scaffolds (the halouracil series, the polyols) look largely
exhausted now. I separately checked fingerprint (Tanimoto, Morgan r=2)
similarity from the current top-10 observed molecules to all unused
candidates: the best available next-nearest-neighbor similarity is only
0.33, meaning there are no further close structural analogs of the current
best hits left in the pool — the halouracil and polyol "veins" are mined
out. This is useful negative evidence: further gains must now come from
the descriptor-level relationship (TPSA/HBD/logP) generalizing to other
scaffolds, not from scaffold-hopping within the current best families.
**Decision:** Continue the same method (ridge surrogate, refit on all 70
points) for exploitation, since it is validated and improving, but also
cross-checked its top picks don't trivially overlap with already-near-
duplicate structures. No restart needed — same features, refit with the
larger dataset (10 more points than the previous fit).
**Hyperparameters:** Same ridge alpha=2.0, same 7 standardized descriptors,
same closed-form fit procedure as the previous step (unchanged; see decision
after evaluation 30). Not re-tuned, since LOO R2 is already improving under
the unchanged configuration — no evidence yet that alpha needs adjusting.
**Reason:** With 205 evaluations remaining, LOO R2=0.84, and clear outperformance
over random sampling already demonstrated, continuing to exploit the
(refit) surrogate ranking is still the best-supported choice. Batch size is
kept moderate rather than spent all at once, because the top-of-ranking
predicted values dropped substantially (the "easy" extreme candidates are
gone), so I want to check whether the next batch's realized values still
beat batch-1's random baseline by a similar margin, or whether returns are
diminishing, before committing further.
**Next step limit:** 50 additional objective calls, at most 205; pause at
cumulative evaluation 120.
**Why this length:** Similar reasoning to the previous block: enough new
data (50 points) to meaningfully re-estimate the surrogate's LOO performance
and detect diminishing returns, but leaves 155 evaluations (>75% of the
remaining budget) for further adjustment if this batch's realized values are
noticeably worse than batch 2's.
**Review:** At evaluation 120, check: (1) realized mean/min/max of this
batch vs batch 2 (-10.40 mean, -20.52 min) to detect diminishing returns,
(2) updated LOO R2, (3) whether new best-so-far improves on -23.62, and
(4) whether predicted-vs-realized values are well calibrated (systematic
over/under-prediction would suggest switching to a nonlinear model, e.g.
k-NN on fingerprints, or reverting to diversity sampling in a fresh region).
No preselected next method.

## Decision after evaluation 120
**Budget:** 120 used / 275 total; 155 remaining.
**Evidence:** Batch 3 (`ridge_surrogate_exploit_v2`, 50 candidates) scored
mean y=-7.12, min=-11.01, max=-3.95 — clearly diminished vs batch 2
(mean -10.40, min -20.52), though still much better than batch 1's random
diversity baseline (mean -4.83, several positive values). No new best-so-far
(still idx 409, -23.62). Predicted-vs-realized correlation within batch 3
alone was only 0.29, noticeably weaker than the LOO R2 would suggest, but
this batch's predictions spanned a narrow band (-9.74 to -6.2), so a modest
absolute error dominates a correlation computed over a compressed range;
RMSE is the more informative diagnostic here.
I used this checkpoint to more rigorously compare modeling options with
n=120 observations (5-fold CV RMSE, since LOO would be slow for tree
ensembles): Ridge-on-7-descriptors=2.00, RandomForest(desc)=2.12,
ExtraTrees(desc)=2.10, GradientBoosting(desc)=2.23, kNN(desc)=2.31,
RandomForest(fingerprint)=3.30, kNN(fingerprint, Jaccard)=4.61,
RandomForest(desc+fingerprint)=3.32 (y stdev=4.48). The simple linear ridge
on the 7 hand-picked descriptors remains clearly the best of everything
tried — fingerprint-based models generalize poorly at this sample size
(too high-dimensional relative to n=120), and tree ensembles on the same
7 descriptors don't beat the linear fit, suggesting the TPSA/HBD/logP
relationship really is close to linear/additive in this range. I also swept
ridge alpha in {0.5,1,2,5,10,20} (CV RMSE minimized at alpha=2, consistent
with the already-chosen value) and tried degree-2 polynomial expansion with
heavier regularization (best polynomial RMSE=2.57 at alpha=50, still worse
than the plain linear fit at alpha=2) — no evidence that added model
flexibility helps at this n. Refit on all 120 points: LOO R2=0.795 (stable,
not degrading). Its new top-40 unused predictions are compressed further
(-6.5 to -5.3), confirming the remaining 389-candidate pool is more
descriptor-homogeneous than what's already been sampled — the method is
reaching diminishing returns on this score, but still clearly outperforms
random selection in expectation.
**Decision:** Continue the same method (ridge, alpha=2.0, 7 descriptors,
refit on all 120 points) — explicitly re-justified here rather than assumed,
per the model comparison above. No switch, since nothing tried beats it.
**Hyperparameters:** Unchanged from the previous two steps: ridge alpha=2.0,
standardized 7-descriptor feature set, closed-form fit. Confirmed via CV
sweep this checkpoint (see evidence) rather than carried over blindly.
**Reason:** No alternative model (RF, ET, GBR, kNN, fingerprint-based, or
polynomial-expanded ridge) beat the current linear ridge under 5-fold CV at
n=120, so there is no basis to switch. Diminishing predicted top-of-ranking
values signal shrinking marginal value per evaluation, which argues for a
smaller next batch (to re-assess more frequently as returns shrink) rather
than a larger one.
**Next step limit:** 40 additional objective calls, at most 155; pause at
cumulative evaluation 160.
**Why this length:** Smaller than the previous two blocks (40 vs 50)
because the signal is visibly weakening (compressed predicted range,
diminishing batch-3 realized mean); a shorter block limits further spend on
a possibly-exhausting vein before reassessing, while remaining large enough
to produce a meaningful refit and a fresh diminishing-returns read.
**Review:** At evaluation 160, check: (1) whether batch-4 realized mean is
close to, better than, or worse than batch 3's -7.12 (continued diminishing
returns vs. a plateau at a positive marginal value over random); (2) whether
new best-so-far improves on -23.62; (3) updated LOO R2; and (4) whether it
is now better value to spend remaining budget on broader diversity
re-exploration (in case there are other unexplored-but-promising structural
clusters the descriptor model underweights) rather than continued narrow
exploitation. No preselected next method.

## Decision after evaluation 160
**Budget:** 160 used / 275 total; 115 remaining.
**Evidence:** Batch 4 (`ridge_surrogate_exploit_v3`, 40 candidates, 7-descriptor
ridge) scored mean y=-6.41, min=-12.74, max=-1.74 — a plateau close to batch
3 (-7.12), not a sharp further decline, but both are well above (better
than) batch 1's random baseline of -4.83. No new best-so-far (still idx 409,
-23.62). Refitting the same 7-descriptor ridge on all 160 points gave LOO
R2=0.740 (down from 0.795 at n=120, and 0.84 at n=70) — a real, not
noise-level, decline — and its new top-40 unused predictions compressed to
-6.05..-4.79, nearly indistinguishable from the random-sample mean. This is
the diminishing-returns signal flagged as a trigger at the previous
checkpoint: the 7 original descriptors (TPSA, HBD, HBA, logP, MW, rot bonds,
arom rings) are running out of discriminating power over the remaining pool.
Rather than default to re-diversifying, I first tested whether a richer
feature set could restore discriminating power, since 160 points can support
more parameters than 30-70 could. I computed RDKit's full descriptor set
(217 descriptors via `Descriptors.CalcMolDescriptors`, NaN/inf zeroed,
near-constant columns dropped by a variance threshold of 1e-6, leaving 177),
and compared 5-fold CV RMSE (seed-averaged over 5 splits) for ridge at
several alpha values: alpha=20 gave 1.96, 40 gave 1.66, 50 gave 1.657
(best), 60 gave 1.66, 70 gave 1.68, 100 gave 1.78 (y stdev=4.48). This is a
clear, non-marginal improvement over the 7-descriptor ridge's CV RMSE of
~2.0-2.1 (R2 roughly 0.74-0.80 depending on fold). I confirmed with LOO CV
on the chosen configuration (177 features, alpha=50): LOO R2=0.845, LOO
RMSE=1.60 — better than every previous checkpoint's figure for the
7-descriptor model, and its new top-40 unused-candidate predictions span a
much wider, more differentiated range again (-9.92 to -5.41) instead of the
compressed band the old model had collapsed to. This indicates the richer
descriptor set is recovering real discriminating signal the 7-feature model
had exhausted, rather than just re-fitting noise (confirmed out-of-sample
via LOO, not just in-sample).
**Decision:** Switch the surrogate from the 7-descriptor ridge to a
177-descriptor ridge (full RDKit `Descriptors.CalcMolDescriptors` output,
near-constant columns removed via variance threshold), still with
standardized features and closed-form/sklearn `Ridge`, still used for
top-ranked exploitation of currently-unused candidates. Observations (all
160 rows) are reused as-is; nothing about the harness or its counter is
reset.
**Hyperparameters:** Ridge alpha=50.0 (selected via 5-fold CV grid search
over {20,30,...,150}, averaged over 5 random fold splits to reduce
fold-assignment noise; alpha=50 was the minimum of the sweep, with 40-60 all
close). Feature set: all RDKit descriptors from `Descriptors._descList`
(RDKit version as installed in this sandbox's `.venv`), filtered by
`sklearn.feature_selection.VarianceThreshold(1e-6)` fit on the observed set
(177 of 217 columns retained), then `StandardScaler` + `sklearn.linear_model.Ridge`
via `sklearn.pipeline.make_pipeline` (inherited defaults for `VarianceThreshold`'s
non-threshold behavior and `StandardScaler`; only `alpha` was tuned for `Ridge`).
**Reason:** The larger, regularized descriptor set gives a substantially and
robustly (via LOO, not just in-sample) better predictive fit than the
7-descriptor model at this sample size, directly reversing the diminishing-
returns trend observed over the last two batches. This is a better use of
the remaining 115 evaluations than either continuing the weakening
7-descriptor model or abandoning exploitation for diversity re-sampling,
since the new model still finds clear structure (wide predicted range) in
the remaining pool.
**Next step limit:** 40 additional objective calls, at most 115; pause at
cumulative evaluation 200.
**Why this length:** Enough to properly test the new, more powerful
surrogate's real-world batch performance (previous batches of this size gave
a clear read on whether a method's gains hold up), while leaving 75 further
evaluations (>35% of total budget) to adapt again if this richer model's
out-of-sample performance disappoints relative to its promising CV numbers,
or to push further if it continues to perform well.
**Review:** At evaluation 200, check: (1) batch-5 realized mean/min/max vs
batch 4 (-6.41 mean, -12.74 min) and vs the new model's predicted range
(-9.92 to -5.41) for calibration; (2) whether new best-so-far improves on
-23.62; (3) updated LOO R2 for the 177-descriptor model; and (4) given only
75 evaluations would remain after this block, whether to spend the rest
purely on continued exploitation or reserve some for a final diversification
pass. No preselected next method.

## Decision after evaluation 200
**Budget:** 200 used / 275 total; 75 remaining.
**Evidence:** Batch 5 (`ridge_bigdesc_exploit_v1`, 40 candidates, 177-descriptor
ridge) scored mean y=-5.38, min=-9.80, max=-2.56 — weaker than batch 4
(-6.41) despite the richer model's promising CV numbers, and only modestly
better than the random baseline (-4.83). Predicted-vs-realized correlation
for this batch was 0.38 (moderate, better than batch 3's 0.29 but far below
the LOO figure) — consistent with a continuing, real trend across all five
batches so far (batch means: -4.83 random, then -10.40, -7.12, -6.41, -5.38
for successive exploitation batches): the pool's remaining low-y candidates
are being steadily skimmed off, so each new top-ranked batch is drawn from a
pool with a higher (worse) true floor than the last. This is expected and
not evidence the method stopped working — the running overall mean across
all 200 evaluations is -6.94, well below the random baseline of -4.83, and
no batch mean has gone above (worse than) the random baseline. No new
best-so-far (still idx 409, -23.62).
I refit the 177-descriptor ridge on all 200 points: 5-fold CV RMSE
(seed-averaged) is essentially flat across alpha in {30..60} (~1.54-1.56,
minimum at alpha=40, alpha=50 close at 1.549) and LOO R2=0.839, LOO
RMSE=1.52 — stable and, if anything, marginally better than at n=160
(R2=0.845 there was similar; both indicate a consistently reliable fit, not
degrading). Its new top-40 unused predictions span -5.59 to -4.23, close to
(slightly better than) the pool's apparent average — i.e. there is less
separation left to exploit, but the model still consistently predicts
below-average y for its top picks, which is the correct, expected signature
of a shrinking-but-still-useful advantage as the best candidates are
consumed.
**Decision:** Continue the same method (177-descriptor ridge, alpha
re-confirmed near 40-50, refit on all 200 points) for one more exploitation
block, given continued (if shrinking) outperformance over random and a
stable, well-validated LOO fit. Not switching to pure diversity re-sampling,
since the model still finds a consistent, validated negative-mean signal
(not noise) in its top-ranked unused candidates.
**Hyperparameters:** Ridge alpha=50.0, same 177-descriptor
(`VarianceThreshold(1e-6)`-filtered RDKit `CalcMolDescriptors` output) feature
set and `StandardScaler`+`Ridge` pipeline as the previous step, unchanged
(re-confirmed via the CV sweep above; alpha=40 was marginally better at this
n but the difference, ~0.002 RMSE, is within fold-assignment noise, so I am
keeping alpha=50 rather than chasing a negligible, noisy improvement).
**Reason:** With 75 evaluations remaining and 309 unused candidates, and the
surrogate still reliably separating better-than-average candidates (LOO
R2=0.84, batch means consistently beating the random baseline), continued
exploitation remains the best-supported choice. Given the shrinking margin,
I am planning this as the second-to-last substantial exploitation block,
reserving a final block to incorporate anything learned here plus wrap-up.
**Next step limit:** 35 additional objective calls, at most 75; pause at
cumulative evaluation 235.
**Why this length:** Smaller than prior full-size batches because the
signal is visibly thinning and I want one more reassessment point before
deciding how to spend the final ~40 evaluations (e.g. continue exploiting
vs. a mixed exploit/diversify split for the last block).
**Review:** At evaluation 235, check: (1) batch-6 realized mean vs batch 5
(-5.38) and vs the random baseline (-4.83) — if batch-6 mean is no longer
reliably below -4.83, continued pure exploitation has lost its edge; (2)
whether new best-so-far improves on -23.62; (3) updated LOO R2; and (4) how
to allocate the final ~40 evaluations. No preselected next method.

## Decision after evaluation 235
**Budget:** 235 used / 275 total; 40 remaining.
**Evidence:** Batch 6 (`ridge_bigdesc_exploit_v2`, 35 candidates) scored
mean y=-4.45, stdev=0.84, min=-7.50, max=-2.69 — now statistically
indistinguishable from (slightly worse than) the original random/diverse
baseline (mean -4.83, stdev 5.54), and no new best-so-far. Looking at the
full per-batch sequence (mean, stdev): diversity-init -4.83/5.54; exploit
v1 -10.40/4.40; v2 -7.12/1.81; v3 -6.41/2.12; bigdesc v1 -5.38/1.73; bigdesc
v2 -4.45/0.84. Two things are now clear: (a) the exploitation edge over
random selection has fully decayed to zero at this point — exactly the
trigger condition set at the last checkpoint; and (b) each successive
exploitation batch has markedly *lower variance* than the original diversity
batch. This matters directly for a minimize objective with (as observed)
a heavy-tailed/outlier-shaped distribution: the single best value found so
far (-23.62) came from the one *diverse*, high-variance batch, not from any
low-variance exploitation batch, and no exploitation batch has come close to
it (-20.52 at best). With the surrogate's informative margin now gone, the
highest-expected-value use of the remaining budget is no longer "pick the
predicted-best few among a shrinking, homogeneous residual pool" but
"maximize the chance of hitting another unseen outlier," which favors
variance (structural diversity) over a now-flat mean-prediction signal.
**Decision:** Switch back to **MaxMin diversity sampling**, but restricted
to the 274 still-unused candidates (so it cannot re-propose anything already
evaluated), using the same method as the initial step (RDKit Morgan
fingerprints r=2/2048 bits, greedy MaxMin picker maximizing minimum Tanimoto
distance to already-picked points within this new batch). This is a
restart of the diversity procedure (fresh seed, operating only over the
remaining pool) rather than a continuation of the ridge surrogate; no ridge
model state needs to be carried over since it is discarded for this step,
but all 235 real observations remain in the harness and are not affected.
**Hyperparameters:** Same fingerprint settings as the initial step (Morgan,
radius 2, 2048 bits); new MaxMin seed=123 (arbitrary, picked fresh since
this is importantly a different candidate universe — the 274 remaining
unused compounds — not the original 509).
**Reason:** With exploitation's margin exhausted and the campaign's single
best result attributable to the only other diversity batch run so far, the
remaining budget is better spent probing structurally distinct, unexplored
regions of the pool for another possible outlier than continuing to draw
near-average picks from an already-thinned, low-variance residual
distribution.
**Next step limit:** 20 additional objective calls, at most 40; pause at
cumulative evaluation 255.
**Why this length:** Deliberately not spending the full remaining 40 at
once: this is a genuine strategy change (back to diversity) being tried for
the first time since evaluation 30, so I want one checkpoint to see whether
it is in fact doing better than the now-flat exploitation batches before
committing the rest of the budget, while still leaving a meaningful 20-call
reserve to react to what this batch shows.
**Why this length (continued):** 20 is large enough to get a readable
mean/min for comparison against both the random baseline (-4.83) and the
flat recent exploitation batches (~-4.45 to -5.38).
**Review:** At evaluation 255, check: (1) this batch's mean/min vs -4.83
(random) and vs recent exploitation batches; (2) whether a new best-so-far
beats -23.62; (3) given only 20 evaluations would remain, decide whether to
spend them continuing diversity, returning to the ridge surrogate refit on
the now-255-point dataset, or some split. No preselected next method.

## Decision after evaluation 255
**Budget:** 255 used / 275 total; 20 remaining.
**Evidence:** The restricted-pool diversity batch (`maxmin_diversity_final`,
20 candidates) scored mean y=-1.81, min=-6.50, max=+2.10 — markedly *worse*
than every previous batch, including the original random/diverse batch
(-4.83) and all exploitation batches (-4.45 to -10.40). No new best-so-far
(-23.62 stands). This disconfirms the hypothesis from the last checkpoint:
restricting MaxMin diversity to the 274 remaining (already-doubly-skimmed,
by both earlier diversity and five rounds of exploitation) candidates did
not find another outlier; if anything it suggests the *remaining* unused
pool is now skewed toward higher (worse) y on average, consistent with both
exploitation and diversity having already both drawn down the genuinely low
values over 255 evaluations.
**Decision:** Return to exploitation for the final block: refit the same
177-descriptor ridge (alpha=50) on all 255 observations and evaluate its
top-ranked candidates among the 254 remaining unused. I checked this refit's
LOO performance before deciding: LOO R2=0.855, LOO RMSE=1.43 — if anything
the best-validated fit of the whole campaign, and its top-20 predictions for
the unused pool (-4.18 to -3.84) are consistently better than the diversity
batch's realized mean (-1.81) and roughly in line with recent exploitation
batches (bigdesc v2 realized -4.45). With no evidence diversity still has an
edge in this now-thinned pool, and good evidence the surrogate still
predicts usefully below the pool average, exploitation is the better-
supported use of the last 20 calls.
**Hyperparameters:** Same 177-descriptor, alpha=50 ridge configuration as
`ridge_bigdesc_exploit_v1`/`v2` (unchanged; reconfirmed via LOO above, not
re-swept, since the previous two checkpoints already established alpha
40-60 are all close and non-material to the outcome at this scale).
**Reason:** This is the last block of the campaign; between the two
options actually tested at this budget level (diversity vs. the validated
surrogate), the surrogate is the only one with current evidence of
real benefit over the pool average, so it is used to spend the remaining
budget.
**Next step limit:** 20 additional objective calls, at most 20; pause at
cumulative evaluation 275 (full budget).
**Why this length:** This is the entire remaining budget — no further
checkpoint is possible after it, so there is no reason to reserve any of it.
**Review:** Not applicable; this is the final step. At its completion I will
record the final outcome and stop.

## Final outcome at evaluation 275
**Budget:** 275 used / 275 total; 0 remaining (budget exhausted).
**Evidence:** The final batch (`ridge_bigdesc_exploit_final`, 20 candidates)
scored mean y=-4.00, min=-6.10, max=-1.45 — better than the preceding
diversity-final batch (-1.81) and roughly consistent with the model's
predicted range (-4.18 to -3.84), confirming the surrogate retained a small
but real edge over the exhausted pool right to the end. No new best-so-far
was found in this final block. Across the whole 275-evaluation campaign, the
best observed value remains **y = -23.62**, at candidate index **409**
(`OC[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)CO`, a hexitol sugar alcohol),
found in the very first MaxMin-diversity batch (evaluation 1 of the
campaign, well before any surrogate was fit). The second-best value found
was y=-20.52 at index 151 (another hexitol-type polyol), found by the first
surrogate-exploitation batch. `harness.save()` was called explicitly at the
end (in addition to the harness's automatic save on budget exhaustion); the
backing socket server process was then shut down.
**Decision:** Stop. Budget is fully exhausted (275/275); no further
`evaluate()` calls are possible.
**Hyperparameters:** Not applicable (no further evaluations).
**Reason:** Hard budget constraint reached.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable.
**Review:** Not applicable — campaign complete.
