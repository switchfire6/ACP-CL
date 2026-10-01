# Independent review of the completed core/residual development cohort

Reviewed after all 168 phases completed on 2026-09-28 UTC. This review uses the
fixed [protocol](../core_residual/development/protocol_at_lock.md),
[summary](../core_residual/development/summary.json), and archived raw phase
records. It introduces no new fit, exclusion, threshold, or promoted control arm.
The summary SHA256 is
`f9de3c5fe1cff7388670dd7395fe4bdb8ce4c6a8e13ce497b7c6533f1b6ba294`.

**The separate architecture fails the prespecified primary screen in both
architectures.** Freezing the learned core while learning a raw-input residual
does not provide the required acquisition/retention balance against joint
training of the same architecture. This is a negative development result for
this recipe, not evidence that useful new representations cannot be learned.
Fresh qualification passes everywhere, and the recurrent feature-adaptation
control supplies one narrower positive finding.

The cohort is complete and eligible: six independent seeds per architecture,
12 shared prefix fits, 36 continuous learned trajectories, 12 fresh references,
552,960 arrivals and 207,360 optimizer updates. Separate and joint have identical
total capacity, initial predictions and learning exposure. They do not have
identical backward work or the same number of trainable parameters.

**Independent arithmetic check.** I reconstructed affected/all-case learning
AUCs, correct-support probe means, the seven numerical primary contrasts and
the feature-attribution contrast per architecture directly from the 168 archived
raw phase records, without importing the scientific summarizer. All 16 paired
statistics, six-seed vectors, sign counts and 20,000-resample bootstrap intervals
match the summary exactly; the maximum scalar discrepancy was zero. Every fresh
qualification group and every primary/attribution Boolean agrees with its fixed
threshold. This is an arithmetic and interpretation check, not an independent
training replication or a replacement for the tensor and portable audits.

All differences below are **separate minus joint**, unless another comparator is
named. Lower Brier is better; higher survival is better. Brier numbers are in
their original units. Survival differences are probability differences, so
-.018595 is a loss of about 1.86 percentage points.

| Fixed primary requirement | Conditional | Recurrent |
|---|---:|---:|
| Complete eligible cohort | Pass | Pass |
| Fresh qualification | Pass | Pass |
| Novel affected Brier AUC <= -.002 | +.016853, fail | +.003917, fail |
| Novel AUC improves in >=5/6 seeds | 4/6, fail | 2/6, fail |
| Novel-end valid-old mode 0 cost <=+.005 | -.010241, pass | +.004605, pass |
| Novel-end valid-old mode 1 cost <=+.005 | -.007719, pass | -.033761, pass |
| Return all-case Brier AUC cost <=+.005 | -.015281, pass | +.033726, fail |
| Post-return novel-law Brier cost <=+.005 | +.014376, fail | -.031541, pass |
| Novel survival AUC difference >=-.01 | -.018595, fail | -.012899, fail |
| Return survival AUC difference >=-.01 | +.004435, pass | -.035136, fail |

The conditional primary AUC difference has descriptive 95% interval
[-.006848, +.042192]; the recurrent interval is [-.000877, +.009242]. Both mean
differences are in the unfavorable direction. Neither interval turns a failed
allocation rule into a pass. Conversely, a mean-based retention guard passing
does not establish statistical noninferiority: recurrent valid-old mode 0 passes
at +.004605 despite interval [-.002303, +.010117], which includes costs exceeding
the +.005 tolerance. Several other passing guard intervals also cross their
practical limits.

**Qualification and representation learning.** The fresh joint architecture
passes both requirements overall and for every physical dependency. The two
numbers in each table cell are terminal marginal-predictor Brier gain and
correct-cue affected Brier benefit, respectively; their thresholds are .02 and
.002.

| Fresh qualification group | Conditional gain / cue benefit | Recurrent gain / cue benefit |
|---|---:|---:|
| Overall, six seeds | .154961 / .075723 | .153525 / .074547 |
| Directional efficiency, two seeds | .159139 / .022669 | .161069 / .026262 |
| Arrival delay, two seeds | .158738 / .008196 | .158058 / .006975 |
| Supply timing, two seeds | .147005 / .196303 | .141446 / .190403 |

Thus the fixed exposure and observations permit learning these dependencies
from scratch in the reference architecture. This reference cannot establish
retention: it has no trained prefix, starts with empty replay and Adam, and is
not the primary continuing-learner comparator.

For feature attribution, separate minus fixed_features novel affected Brier
AUC is -.002489 conditional, interval [-.005594, +.000650], with 4/6 improved
seeds. Its magnitude requirement passes, but its 5/6 consistency requirement
fails. For recurrence the contrast is -.012130, interval
[-.017754, -.007131], with 6/6 improved seeds: the attribution screen passes.
Learning the residual visual frame/temporal encoder helps this recurrent design
relative to fixing that encoder at its random initialization. It does not show
that freezing the core is beneficial, or that the residual learns a privileged
relational representation. It cannot rescue the primary failure.

Recorded separate-arm residual encoder displacements are nonzero in all seeds
(conditional 14.57--35.31, recurrent 33.27--37.17 in the reported parameter L2
measure), while core displacement is zero. These establish parameter adaptation
and preservation, respectively. Displacement alone does not measure useful
representation learning; the controlled feature comparison is stronger evidence.

**Cue and seed heterogeneity matters, but does not authorize exclusions.** The
prespecified three cue groups contain just two seeds each. Their mean primary
novel AUC contrasts are:

| Newly active dependency | Seeds | Conditional | Recurrent |
|---|---|---:|---:|
| Directional efficiency | 16002, 16003 | -.002586 | -.000484 |
| Arrival delay | 16004, 16005 | -.009297 | -.000373 |
| Supply timing | 16001, 16006 | +.062443 | +.012607 |

For conditional learning, the four efficiency/delay seeds improve, while both
supply-timing seeds incur large costs (+.058166 and +.066719). Those two seeds
also lose .071777 and .048431 in novel survival AUC. They are part of the target
problem, not outliers to remove. At novel end, joint conditional cue benefits
are .163098 and .135944 for these seeds; separate gives only .000285 and
-.000184. The problem therefore includes failure to use this newly relevant cue,
despite changes in the residual encoder. It is not adequately described as a
minor retention penalty on otherwise equivalent acquisition.

The recurrent supply-timing deficits are smaller but present in both seeds.
Recurrent novel survival AUC is lower with separate in all six seeds, including
the seeds with favorable prediction-error differences. Better Brier on a subset
does not guarantee better greedy survival. The recurrent feature-adaptation
gain over fixed_features appears in all six seeds and all three cue-group means,
but the pooled primary comparison with joint still fails.

**Retention requires behavioral measurements.** Conditional return Brier AUC
improves in every seed: .093995 separate versus .109277 joint, difference
-.015281 with interval [-.020676, -.010242]. Return survival also improves on
average. This is a real favorable comparison within the study, but comes with
worse novel acquisition and the failed post-return novel guard. A model that
learns a new dependency less effectively can retain older behavior more easily;
the favorable old-condition result alone does not solve the joint objective.

Recurrent return Brier AUC is .153600 separate versus .119874 joint, difference
+.033726 with interval [.014309, .048598]. Five of six seeds have worse return
Brier AUC and all six have lower return survival AUC. Its mean survival cost is
-.035136, interval [-.049357, -.020101]. Frozen core parameters clearly do not
guarantee retention or recovery of the emitted predictions. The residual changes
combined logits; the conditional architecture additionally routes using those
combined predictions. Both predictors depend on the support currently supplied.
Preserved weights, accessible knowledge under correct support, and reliable
online behavior are different properties.

The post-return novel guard must not be mislabeled as a pure forgetting rate.
Conditional separate already has worse correct-support novel error at novel end
(.095845 versus .078518). After return the errors are .103919 versus .089543.
The within-arm increases are +.008074 separate and +.011025 joint. Its failed
+.014376 terminal guard is therefore not evidence that separate forgot more
during the return phase; much of the terminal deficit was acquired earlier.
These within-arm changes are descriptive arithmetic, not a replacement gate.

For recurrence, separate's mean post-return advantage is -.031541, but only
3/6 seeds improve. Seed 16004 contributes -.169303, so the mean must not be
described as a consistent retention gain. Moreover both arms degrade: refreshed
novel error rises from .077366 to .156058 for separate and from .080069 to
.187599 for joint. A relative guard can pass while both absolute outcomes are
poor. The valid-old panels and post-return novel probes use evaluation-only
correct support; they should not be presented as autonomous online recognition
of a returning condition.

There are already architecture-dependent differences during unchanged-law
maintenance: separate minus joint all-case Brier AUC is -.003987 conditional
and +.007586 recurrent. The recurrent cost is present before the new dependency
arrives. The experiment cleanly tests a continuing allocation rule, but it does
not isolate a defect triggered only by novelty.

**Most justified next question:** can an adaptive correction remain available
for a new rule without overriding a useful older predictor in the wrong context,
while still allowing the coordinated adaptation needed for harder dependencies?
This study does not establish that another full-core freeze, extra replay dose,
or longer exposure would answer that question. The joint architecture remains
the reference; no secondary arm becomes the winner.

A small, declared diagnostic of the existing saved endpoints is better justified
before another architecture sweep: compare the complete predictor with its
core-only prediction on identical observations and identical causal support,
then contrast actual-history and already defined refreshed-support evaluations.
Inspect whether the residual correction or combined evidence routing destroys
an otherwise useful core prediction, and whether the conditional supply-timing
failure reflects weak cue-responsive corrections. This would locate an access,
combination, or acquisition bottleneck; it would not itself be a deployable
selection policy or a new scientific pass. A later learning mechanism should
protect applicable **combined predictions**, rather than treat an unchanged
parameter subset as sufficient retention. Whether such protection can coexist
with successful new-rule acquisition remains to be tested under matched controls.

The narrow positive evidence is that raw-input residual representation learning
is possible and useful relative to fixed random features in the recurrent
comparison. The narrow negative evidence is that this fixed division of learning
fails the intended acquisition/retention objective against joint training. Six
seeds, one newly activated dependency per trajectory, one return and physical
horizons 4/8/12 do not establish general continual learning, noise resistance,
autonomous critical periods, or an eventual compounding advantage.
