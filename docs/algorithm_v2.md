# Local critical periods and controlled reopening

The exact implementation is in `plasticity.py`, `controller_v2.py`, `sensor.py`,
and `learner.py`. The original algorithm remains available as `acp`; use
`acp_v2` for this revision. See `experiment_v2.md` for comparisons and limits.

For a feature parameter row, the original multiplier is

\[
m = g_{min} + (g_b-g_{min})(1-c_{eff})^\gamma.
\]

When a unit is actually reset, its incoming row and outgoing column receive a
local multiplier floor for the next \(N\) updates:

\[
f(a,g_b)=g_b+(1-g_b)(1-a/N),\quad a=0,\ldots,N-1.
\]

The effective multiplier is the maximum of the ordinary multiplier and all
incident newborn floors. Row and column floors combine by maximum, not by
addition. Initial model units are not marked as reset newborns. The shared
classifier remains fully trainable. Multipliers act after momentum and on
weight decay; the consolidation anchor force remains separate.

At age `max(newborn_steps, maturity_steps)`, the incoming rows close their
local path-integral window. Positive matured utility increases consolidation;
the first useful anchor copies the trained row rather than averaging it with
the reset initialization. Later anchors use the configured interpolation.
Global consolidation cannot truncate a pending newborn's accumulation window.
Pending newborns cannot be reset again. Recycling still respects incoming and
downstream protection. The CNN replaces only its explicit terminal adapter.

The external drift descriptor pools unaugmented normalized images to 4x4,
flattens them, and applies a private fixed Gaussian projection, fixed bias, and
`tanh`. Vectors use the same fixed nonlinear projection without pooling. Its
weights never learn or reset. Windowed sensor-feature innovations estimate
input drift; the learner's current representation provides health diagnostics.
Training loss and replay damage remain learner-dependent signals.

All v2 variants share a fixed initial developmental schedule. After its close,
the ordinary condition uses the existing sustained novelty, stability,
cooldown, and risk rules. Detected and oracle reopenings both set selected
block gates to 1, followed by existing consolidation-based row attenuation.
The oracle receives only a true signal-domain boundary notification; it has
no domain identity, labels beyond the normal minibatch, or evaluation feedback.
Its pending event expires after three monitoring opportunities or three
monitoring intervals. Events during initial development or an active pulse
are dropped. The oracle therefore tests bounded timing information, not an
optimal policy or an upper bound.

The gain floor is an explicit exception to a contracted block gate, including
one contracted by replay-risk feedback. It can therefore update newborn rows
even while mature rows are contracted. Paired probes measure subsequent
damage; this mechanism offers no guarantee that old predictions are unchanged.

Every update records parameter-weighted effective feature gain, L2 feature
data displacement, and actual resets by population. Yoked controls replay the
exact reset counts/times with their own low-utility selections. The gain-yoked
condition uses one constant equal to the completed v2 run's average effective
feature multiplier. That also changes initial learning and cannot isolate
adult timing alone. Source-trace hashes bind each diagnostic to its source.
