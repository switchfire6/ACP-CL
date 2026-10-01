# Core/residual comparison: antecedents and claim limits

Prepared before this experiment's main fitting, 2026-09-28 UTC. This focused
check places the already specified architecture in context; it is not an
exhaustive novelty review and changes no settings or decision criteria.

The general architectural idea already has close precedents. **Side-Tuning**
adapts a frozen pretrained network by adding a lightweight trainable side
network and combining their outputs through summation. A preserved predictor
with an additive adaptive pathway should therefore not be presented as a new
architectural principle. [Zhang et al., ECCV 2020](https://arxiv.org/abs/1912.13503)

**Progressive Neural Networks** retain earlier columns, add new trainable
columns, and transfer features through lateral connections. Their paper
explicitly identifies parameter growth and choosing the appropriate column
at inference as limitations. These are relevant precedents for the tension
between preserving old computations and allocating capacity for new ones.
[Rusu et al., 2016](https://arxiv.org/html/1606.04671)

**Continual backpropagation** addresses loss of plasticity by selectively
reinitializing a small fraction of less-used units. Its experiments distinguish
declining ability to acquire new tasks from forgetting old tasks. That distinction
matters here: a short acquisition/return comparison cannot demonstrate preserved
plasticity over thousands of changes. [Dohare et al., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7)

## What this experiment asks

Our local implementation combines these motivations with the existing causal
history-conditioned predictors. Both pathways begin within this experiment;
the core learns a prefix from random weights before the residual is allocated.
The residual has raw-input access and initially contributes zero logits. The
conditional variant combines corresponding expert logits before inferring
their applicability from past outcomes. The recurrent variant combines logits
from two independent raw-observation and history encoders.

There is one fixed allocation before a maintenance interval, no later capacity
growth, and no learner-visible change signal at novelty or return. The complete
core Adam state is preserved. The primary control uses the exact same composite
architecture with both pathways trainable, the same input/replay streams, and
the same number of updates. Freezing changes backward work and active capacity.

These are properties of this local design, not an established novelty claim.
The useful result would be a controlled answer about acquisition and retention
in the current relational testbed. It would still need replication, additional
environments and longer sequences before supporting a broader algorithmic claim.

## Two distinctions to retain when interpreting results

1. Fixed core parameters do not imply fixed combined predictions. The residual
   can change the summed logits, and conditional evidence routing can change
   even while the core weights remain identical. Retention must be measured.
2. The fixed-feature control freezes only the residual visual encoder at random
   initialization. Its heads and context/readout still learn. Any attribution
   concerns adapting that encoder over the whole post-prefix trajectory,
   including maintenance; it does not isolate feature changes during novelty
   alone or prove that a specific internal representation has been discovered.

The earlier memory-selection failures and the negative original two-pathway
image experiment remain unchanged. The present study is a separate, explicitly
bounded test of a learned core and adaptive raw-input residual.

## Mathematical scope of the current test

For the recurrent model, before the monotone survival projection, the prediction
for action a and horizon h is

\[
 p_{\theta,\phi}(Y_h=1\mid x,H,a)
 = \sigma\!\left(f_\theta(x,H,a,h)+r_\phi(x,H,a,h)\right).
\]

Here x is the raw observation and H contains only earlier observed actions and
outcomes. Y means both participants persist through the specified horizon.
The residual starts with zero output. The joint arm updates both theta and phi;
the separate arm fixes theta after the shared prefix and updates phi. The
conditional model instead adds corresponding expert logits before computing
evidence weights and mixing the expert probabilities.

All trainable arms minimize the same mixture of performed-action binary
cross-entropy on the incoming packet and a uniformly sampled retained packet:

\[
 L_t=\tfrac12 L_{\mathrm{current},t}
       +\tfrac12 L_{\mathrm{replay},t}.
\]

This loss uses available outcomes, not an inferred universal definition of
improvement. Its practical value is assessed using future *evaluation* outcomes
from known physical conditions; those outcomes do not enter the learner before
prediction. The three physical horizons stay fixed at 4, 8 and 12. Longer
training exposure is therefore not a longer planning horizon.

Freezing theta preserves f's parameters, but does not impose a constraint on
the full prediction's change over old conditions:

\[
 \Delta p_{\theta,\phi}(x,H)\ne0
 \quad\text{is possible even when}\quad\Delta\theta=0.
\]

That is why the prospective decision combines new-dependency Brier improvement
with old-knowledge, return, and newly acquired knowledge retention constraints.
It tests the usefulness of one way of allocating plasticity. It does not equate
parameter persistence with persistence of a learned relationship.
