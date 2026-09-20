# ACP-CL v2 development protocol

This protocol is written before the v2 comparative runs. It is an exploratory
mechanism study, not a preregistration or a confirmatory test. The v0.1 results
and configurations remain archived without alteration.

## Questions and interventions

1. Does a recycled unit need a brief local high-gain period to become useful?
2. Does measuring distribution drift through a frozen input sensor reduce
   self-induced drift from changing learned features?
3. Does selective reopening help when its timing comes from known changes?
4. How much of a difference can reset count and nominal feature gain explain?

`acp_v2` grants reset units a linearly decaying local gain floor, measures drift
through fixed random input features, and consolidates useful newborn units
after maturation. Initial units are not treated as newborn resets. V2 uses a
fixed initial scaffold/open/closing schedule shared by all v2 variants. This
deliberately removes adaptive initial closure from this mechanism test; it is
a revision of v0.1, not an isolated detector substitution. Mature reopening is
adaptive and local. Both detected and oracle reopening use the same unit
amplitude, block selection, replay-risk guard, and bounded duration.

The frozen sensor has no trained weights or labels. Training loss, gradient
conflict, replay damage, and learned-representation health remain endogenous
signals; only distribution-drift features are independent of model changes.

## Conditions

| Condition | Role |
|---|---|
| `er` | Ordinary replay |
| `er_recycle` | Main unrestricted replay plus recycling comparator |
| `acp` | Legacy controller under the same experiment configuration |
| `acp_v2` | Revised learner |
| `acp_v2_no_newborn` | Remove local newborn gain and maturation mechanism |
| `acp_v2_learned_sensor` | Use evolving learner features for drift |
| `acp_v2_no_reopening` | Remove mature reopening |
| `acp_v2_oracle` | Extra-information diagnostic: true signal-domain changes |
| `er_recycle_yoked` | Extra-information diagnostic: exact v2 reset counts and times |
| `er_recycle_yoked_gain` | Same reset schedule, constant feature gain equal to v2's run-average effective gain |

Yoked controls receive the corresponding completed v2 run's allocation trace.
They select their own low-utility reset identities. They match counts per
population and step, not unit identity. Gain matching is parameter-weighted
nominal multiplier matching; gradients, momentum, actual displacements,
consolidation, and compute are not matched. Source-trace hashes are recorded.
Neither yoked controls nor the oracle are valid task-free competitors.
The newborn ablation also changes eligibility and consolidation timing; it is
a mechanism-removal control, not a pure isolated learning-rate contrast.
Newborn gain floors can exceed a contracted block gate, including one reduced
by replay-risk feedback. That exception and its measured damage are recorded;
the replay guard is not a global bound on newborn updates.

## Streams and planned runs

- Gaussian audit: retain the existing eight-experience synthetic task, add v2
  and component/quantity controls, and report replay-buffer nearest-centroid
  accuracy. Its class geometry is an implementation diagnostic.
- Shape development: 16 experiences of 4 fixed shape labels, 8 recurring
  nuisance domains, 20 updates per experience, 32 current and 32 replay items
  per update. All training samples are presented once. Test all conditions.
- Longer shape study: 100 experiences with the same training budgets and
  generator settings. Run replay, replay plus recycling, v2, no-newborn,
  no-reopening, oracle, and both yoked controls.
- Stationary and noise-only controls: 100 experiences, replay plus recycling,
  v2, and no-reopening. Underlying shape-domain changes are absent. Noise-only
  varies bounded sensor noise without changing the signal-domain parameters.

Three paired development seeds (101, 202, 303) are used. Shape labels are fixed
throughout; position, scale, viewpoint, background texture, and class-color
correlations vary. This is a new procedural diagnostic, not an established
benchmark. A low raw-pixel centroid score alone does not establish that it
requires robust shape learning. Follow-up should include decorrelated-color
evaluation and an established continual-learning benchmark.

## Measurements and interpretation

Shape acquisition AUC is measured through the first 256 current-example
presentations of every experience, including the before-training point. The
same fixed horizon is used for per-experience fresh-model references. Replay
presentations are counted separately. Early AUC includes retained knowledge
and transfer, not just new learning. Report the last half of experiences as
the late acquisition endpoint and retain all experience curves.

Full retention evaluation occurs every 10 experiences and at the end; every
acquisition diagonal is measured. Final accuracy covers all seen evaluation
sets. Average incremental accuracy uses complete observed rows only.
Forgetting is a maximum over observed checkpoints and is a lower bound on
forgetting measured at every experience. Pairing uses identical input streams,
model initialization, reservoir membership, and training/probe RNG streams.

Record actual reset counts, newborn/maturation counts, feature-gain budgets,
actual feature data displacement, memory, examples, time, controller states,
reopen timing, and stationary/noise activation. Preserve all attempted runs,
including engineering failures and negative outcomes. No hyperparameter
selection on held-out tests: development runs use validation only.

The strongest interpretation would require v2 to improve retention/acquisition
relative to unrestricted ER plus recycling, with component ablations and
controls supporting the mechanism. Oracle improvement without detected
improvement points toward sensing. Failure of an actively reopening oracle
points toward allocation, block selection, or the task. A never-activated or
vetoed oracle does not test reopening efficacy. No benefit over no-reopening,
or frequent stationary/noise activation, argues against the current trigger.
Small-seed bootstrap intervals are descriptive and unadjusted.
