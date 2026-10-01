# Step D2: recoverable context knowledge under recurrence (the go/no-go study)

**Status.** Prospective protocol, written 2026-09-30 before any D2 fitting,
then revised the same day after an independent critique. The changes are
listed at the end.

**What it follows.**

- The design: [step_d_design.md](step_d_design.md).
- The world: [D0b](../reports/commons_d0b/summary.md).
- The competent core: [D1b](../reports/commons_d1b/summary.md), MB-count.

**The user's framing.** This study decides whether to continue the project,
and whether cloud compute is warranted. The decision rule is declared before
any data.

## Question

Does a **globally consolidated core plus small, evidence-selected context
codes** beat the **strongest amortised in-context learner with the same
modular prior**? It must do so on two things:

- recovering the **rare context**;
- **composing never-seen factor combinations**.

It must do so under an equal memory budget, with the headroom verified by an
oracle-indexed ceiling.

## Environment

- **World:** D0b (key `1586f04b8d3cc8c2`), steady probes, 13 actions.
- **Stream:** 20,000 episodes from the D0 schedule generator:
  - stratified semi-Markov, with run lengths P(L) ∝ 1/L on 8–800;
  - training combinations A0B0C0, A1B0C0, A0B1C0, A0B0C1, plus the
    **minority A1B1C1** (about 5.6%);
  - **held-out combinations** A1B1C0, A1B0C1 and A0B1C1, only in the last
    quarter (30% of it);
  - unannounced switches;
  - common random numbers per (seed, episode).
- **Seeds:** development 22101–22103; test 22201–22212, or 22201–22224 under
  the power rule below.

## Learners

**Shared core (all arms).** D1b's MB-count, with these fixed:

- the Thompson sampling plus count bonus scheme, with c = .05 and **global
  reservoir counts** (a declared fairness choice);
- EMA d = .99, lr .001, U2;
- a failure-hazard ensemble of 5 members (2 × 128 tanh);
- M = 256 planning draws.

**How context enters the models (identical for every arm).** Each arm supplies
three context vectors, c_s, c_p and c_f (6 dimensions each):

| Model | Conditioned on |
|---|---|
| Supply model | mean = table + W_s c_s; log-variance = table + V_s c_s |
| Probe model | The same form, with c_p |
| Failure model | Input gains the concatenation [c_s, c_p, c_f] (18 dimensions) |

The probe model therefore informs decisions only through c_p's role in the
failure model.

| Arm | How the vectors are produced |
|---|---|
| **G**, global | All three are learned constants. There is no context. |
| **AM**, amortised, unfactored | One GRU over the last k episodes' compact summaries (supply sensors, probe sensors, action, failure steps) outputs an 18-d vector, split into the three slots. Trained end to end. |
| **AM-factored**, amortised with the modular prior | Three per-stream GRUs, over supply summaries, probe summaries, and (action, failure) summaries, each outputting 6 dimensions. This matches FC's structural prior. |
| **FC**, factored codes (the hypothesis) | Per slot, a library of at most 6 codes, selected by per-slot evidence (below). |
| FC-single | One library of 18-d codes, selected by all the evidence together. This isolates factoring. |
| FC-K1 | FC with one code per slot, which is never replaced (the library is disabled). It isolates the library from the reparameterised core. |
| **OC**, oracle codes *(privileged)* | FC's architecture, with codes indexed by the true factor level, lagged one episode: A → c_s, B → c_p, C → c_f. It is the indexing ceiling. |
| W, wiped *(descriptive)* | G with all state reset at every true switch. |

**The primary competitor, AM\*.** AM\* is whichever of AM and AM-factored,
each tuned as below, has the lower development whole-stream excess. **Every
criterion compares FC with AM\*.**

**Memory parity.**

- Each AM reservoir entry stores the compact summaries of its preceding k
  episodes. This is what lets AM train end to end.
- AM's reservoir capacity in episodes is reduced so that its total replay
  bytes, including the encoder, equal FC's within 5%.
- FC's bytes count its reservoir, library and per-episode code assignments.
- All bytes are reported.

**The FC mechanism, per slot.**

- **Evidence.**
  - Supply and probe slots: the exact Gaussian log-likelihood of that
    stream's sensors, under the core with each library code and with the
    **default code**.
  - Failure slot: the hazard log-likelihood, scored **only on informative
    episodes**, i.e. those with a 2↔3 transfer. Other episodes carry no
    evidence for this slot.
- **The default code.** It is the running mean of the library codes. The core
  also trains with the default code on 20% of its minibatches, so that the
  default is always in the core's distribution.
- **Filter.**
  - A log-space sticky filter with h = .05.
  - The active code changes only when some code's posterior is at least .95.
  - It spawns a new code, initialised at the default, when the default beats
    every library code by more than τ nats on 2 consecutive informative
    episodes.
  - When the library is full, the code with the fewest assigned episodes is
    replaced.
- **Training.** Code assignments are stored with each replay episode and are
  **never relabelled**. Codes train by gradient only on their assigned replay
  episodes, at the code learning rate. The core trains on all episodes and is
  averaged by EMA.
- No whole-network recall.

## References

- **L**, one-step lookahead. It is the primary reference.
- **O**, which knows the context.
- **I-myopic.**

All three use D0b's tables on every test stream.

## Outcomes

**Two measures.**

- **Counterfactual probes**, which remove exploration noise:
  - taken at episodes {2, 5, 10, 20, 40} after every switch;
  - on a fixed panel of 32 decision states per context, shared by all arms;
  - each arm's greedy action (ensemble mean, no bonus) is scored by O's tables
    over all 13 actions;
  - reported as excess over L's action on the same states.
- **Online regret**, including exploration, scored by O's tables and reported
  as excess over L.

**Windows.** Runs shorter than a window are truncated, and results are
weighted by episodes.

| Window | Episodes after a switch |
|---|---|
| Identification lag | 1–2 |
| Model quality | 3–20 |
| C-specific | Up to 40 |

**Outcomes:**

| Outcome | Measure | Definition |
|---|---|---|
| **R-min**, minority recovery | Probes at {5, 10, 20} | Mean probe excess over every recurrence (not the first encounter) of A1B1C1 |
| **R-held**, held-out composition | Probes at {2, 5, 10, 20, 40} | Mean probe excess on the **first encounter** of each of the 3 held-out combinations |
| **S**, whole-stream | Online | Excess over L across the stream; the late quarter is reported too |
| **Headroom gap** | Probes | G − OC, on the R-min and R-held probes |

**Normalised gain** = (G − arm) / (G − OC): the fraction of the oracle
headroom that an arm captures. It is given for each outcome.

**Supporting reading: knowledge compounding** (declared, not gating).

- **Visit slope.** The regression of recurrence-window probe excess on visit
  index (2nd, 3rd, … visits) per context, for FC and AM\*. Compounding means a
  negative slope: knowledge builds rather than being relearned.
- **Learning to learn.** First-encounter probe excess for the held-out
  combinations (late) compared with the first encounters of training
  combinations (early).
- **Payback.** S over the whole stream against the late quarter.

**Descriptive:**

- the absence-length relation. It is expected to be nearly inert under
  uniform replay, which keeps the minority at about 6% of the reservoir, so it
  is **not a criterion**;
- 2↔3 transfer rates per window and trap flags, per arm;
- the failure taxonomy (stability, plasticity, self-selected-data trap);
- FC diagnostics (codes per slot, code × factor purity, switches, spawns);
- W, G, FC-single, FC-K1, AM and AM-factored;
- costs.

## Pre-declared criteria

**Per-seed pass condition.** A seed passes a comparison when
FC − AM\* ≤ −max(.25 · AM\*, δ), where δ is the development per-seed SD of
the difference, fixed at the lock. A criterion needs that condition in
**≥ 10/12 seeds**, or **≥ 20/24** if the study moves to 24 seeds. It must
also hold on the seed mean.

**Validity checks.** These establish that the study can discriminate.

| # | Check | Rule |
|---|---|---|
| **V1** | Headroom exists | G − OC ≥ .005 on the R-held probes **and** on the R-min probes, in ≥ 10/12 seeds |
| **V2** | AM\* is a strong competitor | (G − AM\*) ≥ .5 × (G − OC) on the R-held probes, seed mean |

**Hypothesis criteria:**

| # | Question | Rule |
|---|---|---|
| **C1** | Not worse overall | S(FC) ≤ S(AM\*) + .002, seed mean |
| **C2** | Better rare-context recovery | R-min, FC vs AM\*, per-seed pass condition |
| **C3** | Better composition | R-held, FC vs AM\*, per-seed pass condition |

## Decision rule (declared now)

| Outcome | Verdict | Project | Cloud compute |
|---|---|---|---|
| V1 and V2 pass; C1, C2 and C3 pass | **STRONG GO** | Continue: a tier-1 standard-benchmark port | Yes, **after a CPU pilot** on that benchmark confirms the setup |
| V1 and V2 pass; C1 passes and exactly one of C2 or C3 passes | **QUALIFIED GO** | One targeted follow-up on CPU | Defer |
| V1 and V2 pass; C2 and C3 both fail, or C1 fails | **MECHANISM NO-GO** | Stop developing the mechanism. The methodology and benchmark may be published as a calibrated negative. | No |
| V1 fails | **UNINFORMATIVE** | There is no indexing headroom, so this world cannot test the hypothesis. Stop this line, unless a specific fix is evident. | No |
| V1 passes, V2 fails | **COMPETITOR TOO WEAK** | A win would mean nothing. Improve AM\* once, in a declared revision, then re-run; otherwise the result is UNINFORMATIVE. | No |

**Two further rules.**

- If the knowledge-compounding reading clearly favours FC (visit slope more
  negative than AM\*'s in ≥ 9/12 seeds) while the verdict is a no-go, this is
  **flagged to the user** for judgement.
- A world-level compounding study (D3) is the first follow-up after any GO.

## Development phase

**Seeds.** 22101–22103.

**Gates, before anything else:**

1. **OC competence gate.** On held-out first encounters, OC's probe excess
   over L must be ≤ .010 from episode 5 on. OC must also beat G (V1-like) in
   ≥ 2 of 3 seeds. If either fails, **OC's implementation is fixed**. This is
   not a verdict.
2. **Core parity.** G on a stationary slice reproduces D1b's MB-count
   competence within .003.

**Declared knobs** (selected on the development whole-stream excess S):

| Arm | Knobs |
|---|---|
| AM and AM-factored | k ∈ {1, 8, 32}; encoder lr ∈ {.001, .003} |
| FC and FC-single | τ ∈ {5, 15}; code lr ∈ {.01, .03} |

AM\* is chosen here.

**Pathology stop (FC).** FC fails if any of these hold:

- more than 3× as many active-code changes as true switches for its factor,
  in any slot;
- more than 6 codes in any slot;
- a mean code-to-factor purity below .6.

**At most one** protocol revision on fresh development seeds is allowed. If
the pathology recurs, the verdict is **MECHANISM NO-GO**.

**Power rule.** From development, estimate the per-seed SD of the C2 and C3
differences. If P(≥ 10/12) at the development effect sizes is below .8 but
P(≥ 20/24) is at least .8, the test uses 24 seeds. This is recorded before
the lock.

## Engineering and integrity

- **Smoke run** (seed 22199, a 2,000-episode stream). It checks:
  - legal observations (only OC sees a label, lagged);
  - W resets exactly at the true switches;
  - byte parity;
  - AM trains on stored summaries only;
  - FC's filter matches a brute-force computation;
  - the default code is in-distribution;
  - informative-episode scoring;
  - determinism and resumption;
  - L and O lookups.
- **Order after smoke:** development, the gates, the lock, the test.
- **Replicate pass.** 3 seeds × {FC, AM\*}, re-fit from scratch and compared
  bit for bit. Every hash and CRC is re-verified.
- **Resources.** At most 6 workers. New files only. Every failure is kept.

**Outputs.**

- `configs/commons_d2{,_smoke,_dev}.json`.
- `scripts/commons_d2.py` and `src/acp_cl/commons/learners_d2.py`.
- `runs/commons_d2*/`.
- `reports/commons_d2/`.

## Changes after the critique (2026-09-30)

1. **The validity check that could not fail.** "Normalised gain ≥ .5 against
   W" is replaced by headroom checks: V1 (G − OC) and V2 (AM\* captures at
   least half of the headroom). Normalised gain is now against G and OC, and
   W is descriptive.
2. **Conditioning, fully specified.** Every slot reaches decisions through
   the failure model. The supply and probe models are parameterised as table
   plus linear code, and G gets the same parameterisation. An OC competence
   gate and a minority headroom check were added, so an implementation bug
   cannot produce a stop verdict.
3. **The structural-prior leak.** The competitor is now AM\*, the better of
   AM and **AM-factored**. AM-factored has the same per-stream modular prior
   as FC.
4. **AM under the memory budget.** AM now stores its histories as summaries
   in its reservoir, charged to its memory budget.
5. **Unambiguous per-seed conditions.** Each is an absolute-or-relative margin
   with δ fixed from development.
6. **Revision cap.** At most one revision, then NO-GO.
7. **Measurement.** Counterfactual probes replace noisy online windows for
   C2 and C3. The windows are split, truncated and weighted, and C3 uses first
   encounters only. There is a power rule to go to 24 seeds.
8. **Absence.** It is now descriptive, because it is inert under uniform
   replay.
9. **AM tuning.** AM gets k ∈ {1, 8, 32} and equal tuning. AM with retrieval
   over the reservoir is noted as a **limitation**; it was not run, to bound
   the cost.
10. **FC fully specified.** The default code, core training on the default,
    informative-episode scoring, τ as a knob, the FC-K1 control, and fixed
    assignments.
11. **Exploration reporting.** 2↔3 transfer rates and trap flags are reported
    per arm, because the global count bonus confounds C.
12. **Labels.** A STRONG GO justifies cloud only after a CPU benchmark pilot.
    A "C5 fails because AM ≈ OC" outcome is now correctly labelled (V2 and
    MECHANISM NO-GO).
13. **Knowledge compounding** was added as a declared supporting reading, at
    the user's request.
