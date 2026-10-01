# Calibrated continual learning: what an exact ideal agent reveals about online learners

*Adam project write-up, covering work from 2026-09-28 to 2026-10-01. Every
claim below links to a prospective protocol, a locked run and an
independently verified report in this repository. Negative results are
reported on the same footing as positive ones.*

The write-up has three parts:

- **Part 1** explains the idea and the results in plain language.
- **Part 2** describes, honestly, what we expected and what we knew was
  already out there.
- **Part 3** onward is the technical detail.

---

## Part 1: the idea in plain language

**The problem.** Most AI systems learn once and are then frozen. The real
world keeps changing, and situations come back. A tool that worked last
winter matters again this winter. A person you haven't seen in a year walks
back in. We wanted a learning system that keeps learning as it goes, and
that, when a familiar situation returns, **quickly recovers what it knew**
rather than relearning from scratch. It need not remember everything
perfectly. It must be able to *get knowledge back* when it matters.

**Our hypothesis.** Our hypothesis was loosely inspired by how brains are
thought to work.

- **A shared core.** Keep one shared, slowly averaged core of general
  knowledge, used in every situation.
- **Small tags.** Add a small "tag" for each kind of situation, capturing only
  what is different about it.
- **Recognition.** When the system notices a familiar situation from what it
  is observing, it pulls up the matching tag.
- **Composition.** Tags for different aspects of a situation should combine,
  so a never-seen *combination* of familiar aspects would be handled well
  too.

**How we tested it.** We built small simulated worlds whose hidden rules
change without warning, and recur.

- **The first world.** Two entities share resources and must both survive.
- **The second world.** Three entities exchange resources through pipes. The
  learner chooses its own actions and sees what happens next.

We could compute **the best possible performance** for each world: an ideal
agent with exactly the same information as the learner. So we could always
say how far a learner fell short, and why.

**What we found:**

1. **Early failures came from learning too slowly and noisily, not from
   missing information.** The ideal agent passed tests the learner failed.
2. **Two standard fixes closed most of the single-situation gap:** replaying
   more past experience, and using a smoothed average of the learner's own
   settings.
3. **Saving a full copy of the learner per situation backfired**, even when we
   told it which situation it was in. Situations share most of their
   structure, and swapping whole copies throws that away.
4. **A learner that picks its own experience can lock itself out of
   knowledge.**
   - In roughly a third to a half of runs, it decided early that one
     important action was bad. It never tried it again, so it never found out otherwise.
   - A small, fading bonus for under-tried actions fixed this every time. It
     paid for itself immediately.
5. **The main test went against our hypothesis.** A simpler competitor beat
   our shared-core-plus-tags design on every measure, and even beat a version
   of our design that was told the true situation. The competitor simply reads
   what happened in the last episode.
   - The reason: in these worlds, one episode's consequences already reveal
     the situation, so stored tags have nothing to add.
   - We then tried to build a world where situations are genuinely hard to
     recognise. That turned out to be impossible in this world family: the
     exact outcomes always give the situation away within a few episodes.

**Bottom line.** Stored, retrievable knowledge does not beat "just read recent
experience" when recent experience reveals the situation. We retired the
mechanism here. The measurement approach, the exploration-trap finding, and
the clean negative result are what this work contributes.

---

## Part 2: what we expected, and what we knew was already out there

We want to be candid about our starting point.

**We knew from the start that the ingredients were not new.**

- The design proposal written before any of these experiments said plainly:
  *"The defensible contribution so far is methodological."*
- It listed close precedents for every part of the idea:
  - fast and slow weights (Hinton & Plaut, 1987) and complementary learning
    systems theory;
  - model averaging (Polyak averaging, EMA);
  - experience replay;
  - inferring context from consequences: COIN, MOLe and CN-DPM, plus the
    project's own earlier conditional learner;
  - shared networks with small context codes: Thalamus-style gating,
    hypernetwork task embeddings and CAVIA;
  - in-context inference in meta-reinforcement learning (PEARL, VariBAD).
- We expected any novelty to come from *testing a particular combination,
  calibrated against an exact ideal agent*, not from a new building block.

**What we expected, and what happened:**

| What we expected | What happened |
|---|---|
| Stored, evidence-selected tags would beat amortised in-context inference in the hardest cases: rare situations, long absences, and never-seen combinations. A pre-implementation critique warned that a strong amortised learner might match the tags, so we made it the primary competitor rather than a strawman. | The competitor won. |
| Getting a competent base learner would be straightforward. | It took three studies. The cause, an exploration trap, surprised us at the time, though in hindsight it is a classic reinforcement-learning problem with classic fixes (optimism, count-based bonuses). |
| Before the final weak-evidence test, we estimated the mechanism's chance of winning at roughly **30–45%**, and said a win would be "partly expected from theory". | The test could not be built in this world family. |

**An independent echo.** An evaluation effort from UC Berkeley (Asawa et al.,
"Continual Learning Bench") reports a result like ours: plain in-context
learning beat more elaborate memory systems.

**The realistic positioning.**

- A solo project is unlikely to out-invent large labs on general continual
  learning. Continual learning in *deployed* models remains widely regarded as
  unsolved.
- What small, carefully measured work *can* offer is precision about **when**
  mechanisms help or fail, measured against an exact ideal. Large-scale
  experiments usually cannot provide that.

---

## Part 3: technical summary

We studied small learners that learn online, from scratch, in simulated
worlds whose hidden rules change and recur without warning. What sets this
work apart is **measurement**. In every study, learners were scored against
**an ideal agent with the same information**, computed exactly or by
verified Monte Carlo. That let us split each learner's shortfall into
information it lacked, data it had not yet seen, and inefficiency in how it
learns.

Main findings:

1. **The failures were caused by learning, not by missing information.** An
   earlier learner failed its acquisition tests. An ideal observer with the
   same inputs passed all 31 of them.
2. **Within a single unchanging context, most of the gap came from single-pass
   online optimisation.**
   - That share was 64% of the excess over the oracle, in 12/12 seeds.
   - Finite data accounted for 21% and pixel perception for 15%.
   - Two standard remedies each closed a large share: full-history replay
     (−37%) and predicting with an exponential moving average of the weights
     (−48%).
3. **Copying a whole network back in for each context hurts, even with a
   perfect context index.** Contexts share most of their structure, and
   swapping whole networks throws that structure away.
4. **Learners that choose their own data fall into exploration traps.**
   - The ensemble agreed early that a key action was bad, never tried it
     again, and so never learned otherwise.
   - This happened in 46% of seeds, and unlimited memory did not help.
   - A fading count bonus removed it in 24/24 seeds (exact McNemar
     p = .001).
   - Cumulative regret was lower in every training block, including the
     first: the exploration paid for itself.
5. **When the recent past reveals the context, amortised in-context inference
   beats stored, retrievable context codes.**
   - Here "amortised in-context" means reading only the last episode.
   - It even beat oracle-indexed codes on never-seen combinations.
   - Our hypothesised mechanism (a shared core plus small codes selected by
     evidence) has **no retrieval headroom** in such worlds.

Several of these findings re-derive known principles: weight averaging,
replay, optimism or count-based exploration, and the strength of in-context
inference. Rediscovering them inside a rig that measures against an exact
ideal is what makes the rig trustworthy. The new contributions are the
calibrated decomposition and the regime-specific answer in finding 5.

## Methodology

**Prospective protocols.** Each study had a written protocol before any
fitting. That covered its questions, arms, knobs, decision rules and what each
outcome would imply. Tuning happened only in declared development phases, on
development seeds. Test seeds were fresh, and configs, protocols and sources
were hash-locked before test runs. Amendments were allowed only before the
lock, and were disclosed.

**Exact references.**

| Reference | What it knows |
|---|---|
| O | The current rule; a context-known oracle |
| I | The rule family but not the current rule; a same-information ideal observer with a sticky Bayesian filter |
| L | I plus one step of lookahead on the value of information. It became the primary reference once it beat a myopic I by 16% of I's regret. |

Each reference was verified against independent brute force, including edge
cases such as clipping, donor-limited transfers, failure before arrival and
negative supply.

**Calibrated bars.**

- Competence bars were checked against achievable ceilings. In one study, a
  feature-level learner showed the bar was reachable at all.
- Validity checks guarded against reaching a verdict for the wrong reason:
  - headroom checks, using an oracle-indexed ceiling and a no-context floor;
  - a competitor-strength check;
  - competence gates on privileged ceilings.

**Integrity.**

- **Determinism and replication.** Fits were deterministic. A from-scratch
  replicate pass compared fits bit for bit.
- **Archive checks.** Every archive was hash- and CRC-verified. This mattered:
  the host machine produced a flipped bit, worker crashes, and one silently
  corrupted file. Each was detected and regenerated identically.
- **Review.** Studies were critiqued before implementation, and their results
  reviewed adversarially and independently.

## Studies and results

### A: information audit and retrieval capacity

**Information audit**
([report](information_audit/README.md)).

- The ideal observer passed every cell the learner failed: Base A Brier .045
  against .196, and every delay cell.
- The learner plateaued at about 2× the ideal error within a single context.

**Retrieval capacity**
([report](retrieval_capacity/summary.md)). **BORDERLINE.**

- Minority-context knowledge was still expressible through one fitted 12-d
  context vector, at .094 against a block-end level of .085.
- The learner's evidence pathway could not reach that vector.

### B and B2: the stationary competence gap

**B** ([report](competence_gap/summary.md)).

- None of the arms was competent.
- The plateau was a steady state: it did not improve from 8k to 32k arrivals.
- Replay capacity narrowly missed its pre-declared threshold (24.1% against
  25%).

**B2** ([report](competence_ceiling/summary.md)). 12 seeds.

- **Decomposition of the gap:** estimation .0078 (21%), perception .0058
  (15%), **online optimisation .0243 (64%)**.
- **Remedies:**

  | Remedy | Excess removed | Seeds |
  |---|---:|---:|
  | Full replay (R256) | 37% | 12/12 |
  | EMA weights | 48% | 12/12 |
  | Both together | 67% | — |

- **EMA lags.** It is much worse in the first chunk (.170 vs .098), and
  better from the third chunk on.

### C: consolidated recall (stopped at development)

([report](consolidated_recall/dev/summary.md))

- **Global averaging under switching** helped modestly: −17% to −19% of
  stream excess.
- **Recall hurt.** Recalling per-context whole-network snapshots was worse
  than plain online learning, **even with oracle indexing**. Each snapshot
  saw only its own context's data, and the shared structure was lost.

### D: a closed-loop resource commons with exact references

**D0 and D0b** ([D0](commons_d0/summary.md), [D0b](commons_d0b/summary.md)).

- **The world.** Three entities exchange resources through channels, with
  agent-chosen transfers and dense sensed consequences.
- **The contexts.** Contexts are combinations of three factors:
  - **seasonality (A)**;
  - **route efficiency (B)**;
  - a **cue-gated delay (C)**.
- **A trade-off.** For factor C, decision relevance and low separability
  could not be had together. The same episodes that make C matter also reveal
  it.

**D1 and D1b** ([D1](commons_d1/summary.md), [D1b](commons_d1b/summary.md)).

- **D1.**
  - A model-based learner, which learns supply and failure models and plans by
    simulation, was nearly competent.
  - It failed through a seed-dependent exploration trap.
  - A model-free learner was about 3× off the bar.
- **D1b.** The count-bonus learner (MB-count):

  | Reading | Result |
  |---|---|
  | Competent | 12/12 seeds |
  | Escaped the trap | 24/24 seeds |
  | Exploration paid for itself | 12/12 seeds |
  | Net cumulative regret against MB | −139 |

  Randomised prior functions did not escape the trap.

**D2: the go/no-go test of the hypothesis**
([report](commons_d2/dev/summary.md)).

- **The competitor.** An amortised in-context learner that reads the last
  episode (k = 1) beat the factored-code library on every measure:

  | Learner | Whole stream | Rare recurrence | Never-seen combinations |
  |---|---:|---:|---:|
  | Amortised, k = 1 | .051 | .033 | .038 |
  | Factored codes | .054 | .052 | .052 |

- **The ceiling.** The amortised learner matched or beat even the
  oracle-indexed codes.
- **Diagnosis.**
  - The context is readable from one episode of dense consequences, so
    retrieval has no headroom.
  - What remains is the core's competence on factor C and on composition.
- **Verdict.** A **development-level mechanism no-go**. The locked test was
  not run, because development showed it had about zero power for the
  hypothesis.

## Limitations

- **Scale.** Small learners and small simulated worlds. The results are
  about mechanisms and regimes, not deployment-scale systems.
- **One world family per question.** D2's negative applies to worlds where
  the recent past reveals the context.
- **No weak-evidence regime here.** A follow-up (D2w,
  [report](commons_d2w/summary.md)) found that this world family has no
  weak-evidence regime. Exact failure outcomes identify every relevant context
  within about 3–6 episodes, whatever the sensor noise. The weak-evidence
  claim therefore stays untested, and would need worlds with noisy or coarse
  outcome feedback.
- **A development-level verdict.** D2's verdict comes from development (3
  seeds), with a diagnosis. It is not a locked test.
- **Not tested here:**
  - world-level compounding (cross-episode carry-over);
  - topology shifts;
  - continual learning in large pretrained models.

## Relation to prior work

Every ingredient has precedents:

- fast and slow weights;
- Polyak and EMA averaging;
- experience replay;
- count-based and optimistic exploration;
- randomised prior functions;
- context inference in meta-reinforcement learning (PEARL, VariBAD);
- contextual memory (COIN);
- model libraries (MOLe, CN-DPM).

An independent evaluation effort (Asawa et al., "Continual Learning Bench")
reports, like D2, that vanilla in-context learning beat elaborate memory
systems.

The contributions here are methodological and regime-specific:

- regret decompositions calibrated against exact same-information ideal
  agents;
- headroom and competitor-strength validity checks;
- a clean demonstration that stored retrievable codes lose when context is
  cheap to read.

## Reproducibility

Each report names its protocol, config, script, locked manifest and raw-run
directory. All locked studies include replicate passes and hash and CRC
verification. The working environment is described in
[docs/ADAM_HANDOFF.md](../docs/ADAM_HANDOFF.md).
