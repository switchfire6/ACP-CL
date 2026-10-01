# What repeated evidence can identify

This explanatory derivation was written while the fixed pilot was running.
It changes no source, threshold, endpoint or decision rule. It specializes a
standard label-corruption argument to the existing simulator; it is not a
claim of a new theorem or an explanation established by the experiment alone.

Let Y_H be physical joint survival at horizon H, conditional on the information
available to a predictor. Write p_H = P(Y_H = 1 | x, a, history). With probability
rho, the simulator replaces the entire reported three-horizon survival vector
using a level sampled uniformly from {0, 1, 2, 3}. The report is 1 for horizon
index j exactly when j < level. Its replacement mean is consequently

    r = (3/4, 1/2, 1/4).

In particular, the terminal replacement mean is 1/4, not 1/2. The replacement
rate is not the realized fraction of wrong individual labels, and the three
reported horizon labels are correlated. The gate therefore uses only the
terminal horizon when accumulating its evidence.

If corruption is independent of the physical outcome conditional on the
declared noise interval, the mean reported outcome is

    q_H = (1-rho) p_H + rho r_H.

This mean can change while p_H stays fixed. Under Brier loss, q_H is the optimal
predictor of those reports. For an ideal physical predictor p_H and an ideal
report predictor q_H, completing the square gives

    E[(p_H - Y_reported,H)^2] - E[(q_H - Y_reported,H)^2]
      = (p_H - q_H)^2
      = rho^2 (p_H - r_H)^2.

Thus sufficiently repeated honest evaluations against corrupted reports can
favor q_H. Yet on the physical target the direction reverses:

    E[(q_H - Y_H)^2] - E[(p_H - Y_H)^2]
      = (q_H - p_H)^2.

This is a population argument assuming the predictors equal their respective
conditional means. Actual finite networks can be miscalibrated, overfit,
forget, and react through corrupted history, so these equations do not predict
the size of the experiment's policy differences. They establish why a report-
based acceptance advantage is insufficient to identify a physical change.
Freezing the candidate prevents training on its own validation observations;
it does not remove bias in the validation target.

The same affine transformation preserves action ranking at a fixed horizon
when rho < 1 and r_H/rho are identical for every action:

    argmax_a q_H(x,a) = argmax_a p_H(x,a).

Therefore an ideal reported-outcome predictor can have worse physical
calibration while choosing the same best action. This is one reason to report
prediction loss and action survival separately. Finite learned approximations,
action-dependent corruption, and uncertain history need not preserve that
ranking. At rho = 1, the reports contain no action preference from p_H.

More generally, the observed conditional law is a composition of physical
dynamics and a reporting channel. Different compositions can give identical
observations. Independent sensors, known restrictions on the reporting process,
controlled interventions, or structural physical assumptions may separate
some possibilities, but each supplies an additional assumption or source of
information. They cannot be silently attributed to repetition alone.

For Adam, the practical question is whether the chosen rule helps under the
declared change/corruption families with only the permitted observations.
The clean evaluator supplies a scientific comparison, not privileged feedback
for the learner. The [related-work review](adam_related_work.md) gives the
broader identifiability and noisy-learning literature, and the
[locked protocol](evidence_consolidation_protocol.md) gives the actual test.
