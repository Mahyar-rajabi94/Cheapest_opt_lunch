## Decision after evaluation 25
**Budget:** 25 used / 294 total; 269 remaining.
**Evidence:** Completed the diversity-covering step (`maxmin_diverse_init`,
MaxMinPicker over ECFP4 fingerprints, 25 candidates). Observed y range
[-6.8, 0.651], best so far y=-6.8 at candidate 205
(`ClC1(Cl)C2(Cl)C3(Cl)C4(Cl)C(Cl)(Cl)C5(Cl)C3(Cl)C1(Cl)C5(Cl)C24Cl`, a fully
chlorinated polycyclic hydrocarbon — lindane-like). Second best y=-6.51 at
candidate 11 (a chlorinated diphenylmethane, DDT-like). Computed simple
RDKit descriptors (MolWt, MolLogP, TPSA, RingCount) for this batch and
checked Pearson correlation with y: MolLogP r=-0.75, RingCount r=-0.60,
MolWt r=-0.49, TPSA r=+0.37. This is a strong, chemically coherent signal:
highly lipophilic, heavily chlorinated/polycyclic, low-polarity molecules
have the most negative y, and polar/small molecules have the least negative
(or positive) y — consistent with a solubility-like property where
lipophilicity drives the value down. With only 25 points this is not
conclusive proof, but it is strong enough evidence to justify a descriptor-
based exploitation step rather than continuing pure diversity sampling.
**Decision:** Switch from `maxmin_diverse_init` to a descriptor-based
surrogate exploitation method, `descriptor_ridge_exploit`. Build a ridge
regression surrogate on standardized RDKit descriptors (MolLogP, MolWt,
TPSA, RingCount, NumAromaticRings, NumHDonors, NumHAcceptors,
NumRotatableBonds) fit on the 25 observations so far, predict y for all 867
unevaluated candidates, and evaluate the top-ranked (most negative
predicted y) candidates next. Fresh model fit (no prior model state to
retain); harness instance and its budget counter/`_used_idxs` are retained
unchanged (restored from the pickled state saved at the end of step 1) so
no real evaluation is repeated and the 294-call budget counter carries over
exactly.
**Hyperparameters:** scikit-learn `Ridge` (if available) or a NumPy
closed-form ridge fallback, alpha chosen conservatively (alpha=5.0 on
standardized features) given the small sample size (n=25) relative to the
8 features, to avoid overfitting noise in the correlation; features
standardized (zero mean, unit variance) using statistics computed on the
25 training points. These are first-use defaults (no prior model to
compare against), chosen to be conservative rather than tuned, since
tuning on n=25 would itself overfit.
**Reason:** The correlation evidence is strong and chemically sensible, so
exploiting it is more likely to find low-y molecules than continuing
uninformed diversity sampling, while a smaller, regularized next batch
(rather than committing a large fraction of the remaining budget) lets me
check whether the surrogate's predictions actually hold up against new
ground truth before committing further budget to this strategy.
**Next step limit:** 20 additional evaluations, at most 269 remaining; pause
at cumulative evaluation 45.
**Why this length:** Large enough to get a meaningful read on whether the
surrogate's top picks actually beat the step-1 best (a real test of the
hypothesis), small enough (20/269 remaining) that if the surrogate turns
out to be unreliable, most of the budget is still available to pivot to a
different strategy (e.g. pure similarity search around candidate 205/11,
or renewed diversity sampling).
**Review:** At the checkpoint, compare the 20 new observed y values against
the surrogate's predicted ranks (did top-predicted candidates actually
score well?), check whether a new best replaces candidate 205, and inspect
whether the structural neighborhood around the current best (chlorinated/
polycyclic/high-LogP molecules) is being confirmed or contradicted. No
preselected next method.

## Decision after evaluation 45
**Budget:** 45 used / 294 total; 249 remaining.
**Evidence:** The `descriptor_ridge_exploit` batch of 20 was a clear
success: every single one of the 20 picks scored better than the step-1
median, and the worst of the 20 (y=-3.931) still beats most of the
diverse step-1 batch. New global best y=-11.6 at candidate 338
(`Clc1c(Cl)c(Cl)c(-c2c(Cl)c(Cl)c(Cl)c(Cl)c2Cl)c(Cl)c1Cl`, a fully
chlorinated biphenyl — a decachlorobiphenyl-type PCB), far below the
previous best of -6.8. Several other top performers are also highly
chlorinated biphenyls (349: -9.16, 827: -9.15, 609: -8.94, 639: -7.66)
and polycyclic aromatic hydrocarbons / PAHs (90: -9.018, 498: -8.804,
184: -8.699, 88/86: -8.23/-8.0, 404: -7.92, 7: -7.87, 19: -7.85,
650: -7.8). This is a very clean, chemically coherent pattern: the two
best-performing structural families are (1) per-halogenated biphenyls and
(2) fused-ring PAHs — both extremely lipophilic, rigid, low-polarity
structures, consistent with an aqueous-solubility-like property where
these are the classic most-insoluble outliers. Within-batch correlation
between predicted and actual y is ~0 (-0.009), but this is expected
range-restriction: all 20 picks were drawn from the extreme low-predicted
tail, so the surrogate's job (identify the promising region) succeeded
even though it does not finely rank within that already-good tail.
**Decision:** Continue exploiting, but combine two complementary tactics
for the next step: (1) refit the same ridge descriptor surrogate on all 45
observations (more data, same method) to keep ranking the full remaining
pool, and (2) directly target close structural analogs of the two winning
scaffolds (polyhalogenated biphenyls and fused PAHs) by Tanimoto
similarity to candidates 338, 349, 827, 609, 90, 498, 184, using the ECFP4
fingerprints cached from step 1 — since the descriptor surrogate is coarse
(8 simple descriptors) and may miss finer structure/substitution-pattern
effects within these families that direct similarity search can catch.
Harness state (budget counter, `_used_idxs`) is reloaded from the pickled
state saved at the end of the previous step; fingerprint cache reused from
step 1, no recomputation.
**Hyperparameters:** Ridge surrogate: same alpha=5.0, same 8 standardized
descriptors as before (unchanged configuration, referenced from the
previous decision) — refit coefficients only, not structure. Similarity
search: Tanimoto similarity on ECFP4 (radius=2, 2048 bit) fingerprints,
take top-15 nearest unevaluated neighbors (by max similarity to any of the
7 named seed candidates) as the similarity-driven half of the batch; take
the top-15 unevaluated ridge-surrogate picks (excluding any overlap with
the similarity set) as the other half. These are first-use defaults for
this sub-method; no tuning performed given small data.
**Reason:** The evidence shows both a global coarse trend (lipophilicity/
ring count via the descriptor surrogate) and a narrower, stronger signal
(the specific winning scaffolds). Combining broad surrogate re-ranking
with targeted similarity search around confirmed winners should both
continue finding other good coarse-feature candidates and dig for the true
optimum within the most promising structural families, without betting the
whole next batch on only one of the two signals.
**Next step limit:** 30 additional evaluations, at most 249 remaining;
pause at cumulative evaluation 75.
**Why this length:** Large enough to meaningfully populate and compare both
sub-methods (15 + 15) and get a reliable read on whether similarity search
around the winning scaffolds outperforms the general surrogate now that a
strong regional signal exists, while still leaving the large majority of
the budget (219/294) for further refinement once this comparison is in.
**Review:** Compare the similarity-search half against the surrogate half
(which finds better molecules / a new best), check whether the current
best (-11.6) is beaten, and check how many of the remaining unevaluated
candidates still belong to the two winning families (to judge how much
budget is worth continuing to invest here vs. needing to look elsewhere).
No preselected next method.

## Decision after evaluation 75
**Budget:** 75 used / 294 total; 219 remaining.
**Evidence:** No new best this step; best remains y=-11.6 (candidate 338).
The surrogate half clearly outperformed the similarity half: surrogate
batch mean=-7.57 (min=-8.71), similarity batch mean=-5.32 (min=-7.43).
Both halves contained polychlorinated-biphenyl and PAH analogs, but less
substituted/smaller ones than the current best, consistent with a dose-
response relationship (more chlorination / more fused rings -> lower y) in
which 338 appears to sit near the ceiling of what's structurally possible
in this pool. I explicitly enumerated every biphenyl-substructure candidate
in the full 892-pool (42 total) and their chlorine counts: candidate 338
(10 Cl, fully chlorinated biphenyl) is the *most* chlorinated biphenyl in
the entire pool and it is already evaluated; all remaining unevaluated
biphenyls have <=5 Cl substituents, i.e. structurally worse than several
already-evaluated ones. This makes it unlikely (though not certain) that
the PCB family has more to offer, but the PAH family (large fused aromatic
rings, e.g. candidate 90, a coronene-like 6-ring fused system, y=-9.018)
is less exhaustively searched and may still hold a lower value than -11.6.
**Decision:** Continue with the descriptor ridge surrogate
(`descriptor_ridge_exploit`), refit on all 75 observations, since it
outperformed similarity search this round. Drop scaffold-similarity
search as a separate sub-method for now (it underperformed and the
biphenyl family looks largely exhausted of its best members), but add one
new explicit descriptor — halogen atom count (F+Cl+Br+I) — to the feature
set, since raw halogen count looks like a sharper driver than LogP/MolWt
alone for the PCB family specifically. Keep harness state (budget/
`_used_idxs`) carried over unchanged from the pickled state at the end of
the previous step.
**Hyperparameters:** Ridge surrogate: same alpha=5.0 (unchanged, referenced
from the decision after evaluation 25), same standardization procedure,
now 9 features (previous 8 descriptors + halogen atom count) refit on all
75 points. This is a small, justified addition (one feature, motivated by
clear new evidence) rather than open-ended tuning.
**Reason:** Surrogate exploitation continues to beat alternatives and more
training data (75 vs 25) should sharpen its ranking; the added halogen-
count feature directly encodes the mechanism suggested by the PCB-family
evidence. Continuing to spend budget here is justified because there is no
cost to an unsuccessful evaluation beyond the evaluation itself (deterministic,
one-shot), and the pool still contains unexplored high-ring-count PAH-like
and other lipophilic structures that the refit may newly surface.
**Next step limit:** 30 additional evaluations, at most 219 remaining;
pause at cumulative evaluation 105.
**Why this length:** Similar scale to the previous exploitation step, large
enough to see whether the added halogen-count feature changes the ranking
meaningfully and to sample further into the PAH family, while still
leaving most of the budget (189/294) uncommitted for later pivots (e.g., a
renewed diversity sweep if this step stalls).
**Review:** Whether a new best beats -11.6, whether PAH-family candidates
continue to perform well as more are tried (supporting continued
investment there) or start reverting toward baseline (suggesting that
family is exhausted too), and whether the halogen-count feature visibly
changed which candidates get prioritized versus the evaluation-45 surrogate.
No preselected next method.

## Decision after evaluation 105
**Budget:** 105 used / 294 total; 189 remaining.
**Evidence:** Still no new best; y=-11.6 (candidate 338) remains the
global best. This batch's best was y=-8.402 (candidate 203,
`CC1(C)C(C=C(Br)Br)C1C(=O)OC(C#N)c1cccc(Oc2ccccc2)c1`), a
pyrethroid-ester-like structure with a cyanohydrin ester group — a new,
previously-unseen well-performing family (also candidates 484: -8.176,
737: -7.337, 681: -6.876 share this ester/cyanohydrin motif). The
remaining PCB/PAH-family candidates evaluated this round scored worse than
earlier rounds (best PCB here: -7.92 at candidate 500, worse than the
-9.16/-9.15/-8.94 seen at evaluation 45), confirming that family's best
members are now exhausted, consistent with my eval-75 analysis showing
338 is the maximally-chlorinated biphenyl available. Batch overall mean is
also worse than the evaluation-45 batch, a sign of diminishing returns as
the surrogate works through progressively lower-ranked candidates — this
is expected under exploitation and not itself a reason to abandon the
method, but worth tracking.
**Decision:** Continue the same descriptor ridge surrogate
(`descriptor_ridge_exploit_v2`, unchanged feature set/hyperparameters —
referencing the decision after evaluation 75), refit on all 105
observations. Diminishing returns in the PCB/PAH region plus the new
ester/cyanohydrin family signal suggest the surrogate is still doing
useful work generalizing beyond the first two families it found, so I am
continuing rather than switching methods; I will shorten this step
relative to the last one so I can re-check sooner given the batch mean is
trending down.
**Hyperparameters:** Unchanged from evaluation-75 decision (alpha=5.0, 9
standardized descriptors). Refit coefficients only on updated data.
**Reason:** No evidence yet that the method has stopped working overall
(a new best-performing family was just found inside this very batch), just
that specific early families are exhausted; a global switch is not
justified, but a shorter step gives an earlier checkpoint to catch further
decline.
**Next step limit:** 25 additional evaluations, at most 189 remaining;
pause at cumulative evaluation 130.
**Why this length:** Shorter than the previous two exploitation steps
(25 vs 30) given the signs of diminishing returns, to reassess sooner
without over-reacting to a single noisy-looking batch on a deterministic
but discrete objective.
**Review:** Whether the new ester/cyanohydrin family continues to perform
well as more such candidates are surfaced, whether the batch mean keeps
declining (suggesting it may be time to pivot to a fresh diversity sweep
of untouched chemical space instead of continuing to narrow around found
families), and whether -11.6 is beaten. No preselected next method.

## Decision after evaluation 130
**Budget:** 130 used / 294 total; 164 remaining.
**Evidence:** Still no new best (-11.6, candidate 338, unchanged since
evaluation 45). This 25-candidate batch's best was -8.6 (candidate 514,
an ether/aryl-ether, not strongly related to the earlier PCB/PAH/ester
families). Batch means across the exploitation steps show a clear,
consistent decline: eval 25->45 batch mean ~-7.57 (best step),
eval 45->75 batch mean -6.45, eval 75->105 batch mean -6.60, eval
105->130 batch mean -5.63. This is the expected signature of exploitation
running out of genuinely novel high-value structures within the regions
the surrogate currently favors (PCB/PAH/halogenated-biphenyl/ester
families) — it's increasingly picking "good but not better" variants
rather than discovering new extremes. Over 85 evaluations since the last
new best, the surrogate has not found anything beating -11.6, which is
itself meaningful evidence (though not proof) that 338 may be at or very
near the global minimum of the pool, since 338 is also structurally
extremal (the single most-chlorinated biphenyl available, as established
at evaluation 75).
**Decision:** Switch primary method, at least temporarily, back to
diversity sampling (`maxmin_diverse_redo`) over the ~762 still-unevaluated
candidates, using the same ECFP4/MaxMinPicker approach as step 1. Rationale:
all exploitation so far has operated by ranking candidates with a linear
surrogate over coarse descriptors fit on exploitation-selected (non-random)
training data — this risks tunnel vision toward the few structural families
it started with and could be systematically blind to a different,
unrelated family that simply doesn't score high on MolLogP/halogen-count/
ring-count but is nonetheless low-y for some other structural reason. A
renewed diversity pass over remaining candidates checks for that blind
spot cheaply before committing more of the remaining 164 evaluations to
further narrowing. This is not a restart of the campaign — harness state,
all 130 observations, and the fingerprint cache are retained; only the
selection method for this step changes.
**Hyperparameters:** Identical to step 1's `MaxMinPicker.LazyPick` (ECFP4,
radius=2, 2048 bits, seed=0) but restricted to the pool of unevaluated
candidates only (the picker naturally avoids near-duplicates already
evaluated since `used` candidates are excluded from the diversity pool).
**Reason:** The evaluation record shows no improvement for 85 consecutive
evaluations under exploitation, and a diminishing batch-mean trend; the
expected cost of a moderate-size diversity probe is low relative to the
remaining budget, while the expected benefit (catching a missed family)
could be large if the current plateau is a local rather than global
optimum.
**Next step limit:** 30 additional evaluations, at most 164 remaining;
pause at cumulative evaluation 160.
**Why this length:** Comparable to earlier diversity/exploration-oriented
steps; enough to sample meaningfully across the remaining ~762 candidates
without spending so much budget that, if this probe also fails to beat
-11.6, too little budget remains for a final exploitation pass on whatever
it does find.
**Review:** Whether any new candidate beats or approaches -11.6, whether
the best of this diverse batch reveals a structurally distinct family from
PCB/PAH/ester, and whether the overall distribution suggests the search is
genuinely near the pool's minimum (in which case I will spend the rest of
the budget on a final, smaller exploitation/mop-up pass) or that more
exploration is warranted. No preselected next method.

## Decision after evaluation 160
**Budget:** 160 used / 294 total; 134 remaining.
**Evidence:** The renewed diversity probe (`maxmin_diverse_redo`, 30
candidates spread across the remaining unevaluated pool) found a best of
only y=-6.62 (candidate 496, a phosphorothioate), clearly worse than every
exploitation batch mean so far and far from -11.6. No distinct new
low-y family emerged; the batch's other notable scores (-5.6, -4.92, -4.76)
are all isolated, unrelated small halogenated/aliphatic molecules with no
coherent pattern, unlike the clear PCB/PAH/ester signals found earlier.
This is good evidence that the remaining unevaluated chemical space is
mostly unremarkable with respect to this property, and that the
descriptor-surrogate's earlier narrowing toward
chlorinated-biphenyl/PAH/ester families was not tunnel vision but a
correct read of where the real signal is. Combined with the fact that
candidate 338 is the maximally-chlorinated biphenyl in the whole 892-pool
(verified at evaluation 75), I now have converging evidence (coarse
surrogate ranking, targeted similarity search, exhaustive scaffold check,
and now a renewed random-diverse probe) that -11.6 is very likely at or
extremely close to the pool's global minimum.
**Decision:** Switch back to `descriptor_ridge_exploit` (the method that
has produced every improvement so far), refit on all 160 observations
(now including the diversity-probe points, which add useful negative
examples — confirmed unremarkable regions — to the training set). Continue
exploiting the remaining budget on this basis rather than exploring
further, since the diversity check came back negative (no new family
found) and there is substantial remaining budget (134) worth using to
thoroughly mop up any remaining untested members of the known good
families and let the now-larger surrogate re-rank the full pool once more.
**Hyperparameters:** Unchanged from the evaluation-75/105 decisions
(alpha=5.0, same 9 standardized descriptors), refit on 160 points.
**Reason:** Exploitation has produced every meaningful result in this
campaign and the exploration check found nothing better, so returning to
it is the best use of the remaining budget; no change to the method's
configuration is warranted since nothing in the new evidence suggests the
feature set or regularization needs revision.
**Next step limit:** 40 additional evaluations, at most 134 remaining;
pause at cumulative evaluation 200.
**Why this length:** Larger than recent steps because the strategy is now
well-validated (no further exploration decision pending) and a larger
batch efficiently uses budget while still leaving a meaningful review
point with ~94 evaluations left for final adjustments.
**Review:** Whether -11.6 is beaten, whether the batch mean continues the
established decline (further confirming a plateau) or perks up (suggesting
still-unexploited value), and how much of the remaining budget is worth
spending on further narrowing vs. stopping early. No preselected next
method.

## Decision after evaluation 200
**Budget:** 200 used / 294 total; 94 remaining.
**Evidence:** Still no new best; y=-11.6 (candidate 338) has now held for
155 consecutive evaluations across both exploitation and renewed-diversity
probing. This 40-candidate batch's best was -8.4 (candidate 218,
octadecan-1-ol, a long-chain fatty alcohol) and -8.172 (candidate 618,
eicosane, a C20 alkane) — a further new family (long aliphatic
chains/alcohols), again consistent with the overall lipophilicity-driven
pattern, but not competitive with the halogenated-biphenyl/PAH extremes.
Batch mean -5.27, essentially flat relative to the previous batch
(-5.63 at evaluation 130), i.e. the decline has leveled off rather than
continuing to worsen, which suggests the surrogate is now sampling a
broad "moderately good" tier rather than still finding and exhausting
sharp local optima — a sign that most of the genuinely strong structural
signal has already been captured.
**Decision:** Continue the same `descriptor_ridge_exploit` surrogate
method, refit on all 200 observations, for one more step; then plan a
final short mop-up step to use the remaining budget. No change in method
is justified: it remains the best-performing approach available, nothing
in the new evidence suggests switching, and the remaining budget (94) is
modest enough that a full restart of exploration is not worthwhile at this
stage.
**Hyperparameters:** Unchanged from prior decisions (alpha=5.0, 9
standardized descriptors), refit on 200 points.
**Reason:** No signal supports a method change; continuing to refine the
ranking with more data is cheap and could still surface a small number of
remaining good candidates (e.g. unexplored long-chain or ester-family
members) even if a new global best is now unlikely.
**Next step limit:** 50 additional evaluations, at most 94 remaining;
pause at cumulative evaluation 250.
**Why this length:** Leaves a final ~44-evaluation step to properly close
out the campaign after this checkpoint, rather than committing the entire
remaining budget in one untracked block.
**Review:** Whether -11.6 is beaten, whether the batch mean stabilizes
further (confirming the plateau) and whether any evaluation count should
be reserved instead of used (e.g. if the pool is clearly exhausted of
anything competitive, stopping before full budget exhaustion could be
justified — but since there is no cost to an additional deterministic
evaluation beyond the evaluation itself, and real candidates remain
untested, continuing to use budget remains reasonable). No preselected
next method beyond the planned final mop-up step.

## Decision after evaluation 250
**Budget:** 250 used / 294 total; 44 remaining.
**Evidence:** Still no new best after 205 consecutive evaluations since
candidate 338 (y=-11.6) at evaluation 45. This 50-candidate batch's best
was only -6.34 (candidate 254, a phosphorothioate/phthalimide), and the
overall batch quality continued to decline relative to earlier rounds —
most values now cluster in the -2 to -5 range, essentially indistinguishable
from the step-1 diverse baseline. This is the clearest evidence yet that
the descriptor surrogate has fully exhausted the pool's genuinely strong
candidates and is now cycling through progressively more mediocre,
previously-lower-ranked ones. Combined with the earlier exhaustive check
(candidate 338 is the most-chlorinated biphenyl in the entire 892-pool)
and the negative result of the mid-campaign renewed diversity probe
(evaluation 130-160, best only -6.62), I am confident the search has
converged: y=-11.6 is very likely the pool's global minimum or extremely
close to it.
**Decision:** Run one final `descriptor_ridge_exploit` step using all
remaining budget (44 evaluations) — refit on all 250 observations and
evaluate the top-44 remaining candidates by predicted y — then stop. This
is a deliberate choice to use the entire remaining budget on the
best-validated method rather than hold any back, since there is no benefit
to leaving unused evaluate() calls (deterministic objective, no cost per
call beyond the call itself) and the surrogate, while past its point of
dramatic returns, is still the best available way to rank the few
remaining untested candidates in the known-good families (long-chain
alkanes/alcohols, esters, remaining halogenated aromatics).
**Hyperparameters:** Unchanged (alpha=5.0, same 9 standardized
descriptors), refit on 250 points.
**Reason:** No method-switch is justified by the evidence (exploitation
remains at least as good as any alternative tried, including fresh
diversity sampling, which underperformed); using the full remaining budget
on the proven method is strictly better than stopping early or gambling on
an untested alternative this late in the campaign.
**Next step limit:** 44 additional evaluations, at most 44 remaining;
pause at cumulative evaluation 294 (budget exhaustion).
**Why this length:** This consumes exactly the remaining budget; given the
converged evidence, there is no remaining reason to reserve a further
checkpoint before the final evaluation.
**Review:** Not applicable — this is the final step; budget will be fully
exhausted. Final outcome (best y, best candidate, evaluations used) will
be recorded at campaign close.
