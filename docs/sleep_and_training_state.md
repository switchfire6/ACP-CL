# Sleep, pruning, and training-state resets

A functional analogy is worth investigating: recalibrate the processes that
control future learning while preserving useful learned structure. Our current
experiment separates two engineering operations; it does not simulate sleep.

| Operation | What this code changes | Limit of the analogy |
|---|---|---|
| Adam reset | Clears gradient moment accumulators and update counters; keeps weights. | These optimizer statistics have no established one-to-one biological counterpart. |
| Replay reset | Discards example packets and restarts their reservoir bookkeeping; the buffer refills. | Sleep can reactivate experience; clearing the buffer does not implement that process. |
| Synaptic pruning | Not implemented: no connections are removed. | Discarding stored examples is different from changing network structure. |

In mouse motor and sensory cortex, de Vivo and colleagues found smaller
axon–spine interfaces after sleep, with selective scaling that spared some
large synapses. This supports a synaptic-renormalization hypothesis; it is not
evidence that the brain zeros an optimizer or erases its experience memory.
[Primary study, Science 2017](https://pubmed.ncbi.nlm.nih.gov/28154076/).

Giri and colleagues recorded rat hippocampal activity and found that sleep
deprivation diminished reactivation and replay of waking experience, despite
continued sharp-wave ripple activity. This supports treating replay as an
active part of the sleep analogy, rather than equating sleep with an empty
replay store. [Primary study, Nature 2024](https://pubmed.ncbi.nlm.nih.gov/38867049/).

An engineering hypothesis suggested by these findings would combine selective
retention/rehearsal with recalibration of adaptation. It would need a separate
bounded-compute experiment that matches extra updates, preserves useful old
predictions, and shows renewed learning. Neither a benefit from Adam reset nor
a benefit from clearing replay by itself validates that hypothesis.

For now, the optimizer-by-replay factorial diagnoses which carried state causes
the observed acquisition/retention tradeoff. A sleep-inspired intervention is a
possible later experiment, not a biological interpretation of these controls.
