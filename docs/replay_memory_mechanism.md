# Adam: what the two memory timescales mean

This note explains the fixed replay policies in the
[prospective experiment](replay_renewal_protocol.md). It is a description of the
implemented sampler, not a claim that this is a novel algorithm or that its
learning effects are already established.

The learner sees a stream of completed experience packets. Each packet binds
recent, past-only observations and feedback to a later query and its observed
outcome. A stored packet therefore rehearses a conditional prediction problem.
The policy changes which of these problems is revisited; it does not change
the neural architecture, supply a task identifier, or add an optimizer step.

Let t be the number of packets already observed, M=16 the total capacity, and
k=8 the recent capacity. Once both stores are full, the marginal probability
of replaying a particular packet i under the split policy is

    P(replay i) = 1/M                         if i is among the most recent k,
                  (M-k) / [M (t-k)]          otherwise.

This marginal averages over reservoir-membership randomness. In a particular
realized buffer, a stored packet has probability 1/M per draw and an absent
packet has probability zero. An old packet has probability (M-k)/(t-k) of
being stored. The recent and historical stores have disjoint packet IDs; an
evicted recent packet becomes eligible for the historical reservoir exactly
once. Adjacent packets may naturally share raw support/query observations.

With M=16 and k=8, half of replay draws go to the latest eight packets in
expectation, while half go to a uniform sample of older packets. Ordinary
reservoir replay instead gives every past packet marginal probability 1/t.
Recent-only replay gives the latest 16 packets probability 1/16 each and all
older packets zero. The capacity and one-replay-per-update rule stay equal.

Define a completed packet's age as t-1-i, so the newest stored packet has age
zero. After t>=16 packets, the expected mean stored age is:

    uniform:      (t-1)/2
    recent-only:  7.5
    split 8+8:    t/4 + 3.5

These are expectations over the sampler, not observed learning results.
Sampling before the next packet is added preserves causality. A packet contains
32 query arrivals in the main study, but packet age is an exposure count, not
biological time or wall-clock duration.

Why distinguish deletion from age? In a full uniform reservoir, the next
packet replaces an entry with probability M/(age+1). Deleting packets changes
what can be rehearsed immediately. Rebasing effective age changes subsequent
replacement probabilities. A nonempty reservoir must be rebased to its current
occupancy, rather than zero. Resetting random generators changes the realized
draws. The decomposition tests these operations separately at one introduction;
it does not choose the continuous policy's settings.

The scientific question is whether concentrating some rehearsal on recent
relationships can improve acquisition while the historical part preserves
useful predictions through return, revision and noisy reports. A smaller mean
replay age is not itself an improvement: predictive error, retained knowledge,
and survival must meet the declared requirements. The weights also carry
history beyond what the explicit buffer stores.
