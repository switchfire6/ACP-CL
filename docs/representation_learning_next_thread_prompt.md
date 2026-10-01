# Prompt for the next research thread

Paste the following into a new Codex/ChatGPT research thread:

> I am continuing the **Adam** continual-learning research project in
> the repository root. The long-term objective is a general
> learning mechanism that can learn new representations from scratch under
> changing conditions. The image simulator is a testbed, not the final target.
> The motivating idea is that useful learning concerns relationships that help
> entities persist under uncertain environments; environment/context can recur
> without being announced, and perfect retention may be less useful than
> recoverable, adaptable knowledge.
>
> First read these files before proposing or running anything:
>
> - `docs/ADAM_HANDOFF.md`
> - `reports/representation_learning_results.md`
> - `docs/representation_learning_design.md`
> - `docs/representation_learning_protocol.md`
> - `docs/research_synthesis.md`
> - `reports/acquisition_results.md`
> - `reports/training_state_results.md`
> - `reports/causal_access_results.md`
>
> Current state: we implemented a prospective temporal-reconstruction design,
> but its required outcome-only qualification failed before the reconstruction
> arms were run. Separate fresh fits learned all three temporal dependencies;
> five-law blocked interleaving failed four fixed checks: base-A endpoint Brier
> was .195597 against a .12 ceiling, and the arrival-delay cue was unavailable
> at all three cumulative stages. The candidate sequence-reconstruction method
> is therefore **untested**, not disproven. The fixed main comparison must not
> be run or rescued by tuning alpha/exposure, dropping the delay cue, or changing
> thresholds. Completed historical branches, especially frozen-core/additive-
>residual causal access, are also stopped and must not be reopened by a sweep.
>
> Please begin with an evidence-based interpretation of this result. Distinguish
> what it demonstrates from what it does not. Then recommend the most promising
> next bounded research question. I am especially interested in whether the
> bottleneck is context/environment identification, sparse outcome feedback,
> finite history, or representation learning itself, and whether a new design
> could better match the original relational/persistence idea without giving the
> candidate privileged latent state or counterfactual outcomes.
>
> Do not launch new scientific fitting immediately. First propose a small,
> falsifiable diagnostic or a distinct prospective experiment with matched
> information, memory, and compute across comparators. Preserve all locked
> sources/runs/protocols and existing results. Be candid if the evidence suggests
> a larger benchmark change or a conceptual pivot is warranted rather than more
> tuning of this testbed.
