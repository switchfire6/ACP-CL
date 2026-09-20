# Procedural shapes: color controls

The optional `data.color_policy` selects how foreground palette choice relates to the fixed four shape labels. It changes rendering only. The learner still receives an image tensor and its shape label; domain identities, color settings, and curriculum boundaries remain evaluator metadata.

| Policy | Training color correlation | Validation and test correlation |
|---|---|---|
| `original` (default) | Existing domain-specific setting | Existing domain-specific setting |
| `independent` | 0 throughout | 0 throughout |
| `early_biased` | 0.95 for the first `bias_experiences`, then 0 | 0 throughout |

`bias_experiences` defaults to 8 and is valid only with `early_biased`. It must be a nonnegative integer. Zero makes all rendered tensors identical to `independent` under the same seed and configuration. Values longer than the stream are permitted for short smoke runs; such runs contain no later independent-training period. Policy strings are exact and case sensitive.

The correlation parameter is the probability of selecting the label's preferred palette entry instead of a uniform random entry. At 0.95, the preferred entry appears with probability `0.95 + 0.05 / 4 = 0.9625`. At zero, every palette entry has probability one quarter independently of the label. This changes a shortcut while retaining the original fixed shape target.

Both new modes consume an association draw and a uniform palette draw for every rendered sample. Switching between them therefore preserves position, scale, rotation, silhouette coverage, texture draws, color jitter, sensor noise, and subsequent RNG state. Only the selected foreground palette entry changes during the biased training period. Pixels outside the silhouette are identical; foreground differences follow the palette change, subject to clipping and uint8 rounding. Validation/test tensors and all training tensors after the bias period are exactly identical across the two curricula when seeds and other settings match.

The default retains version-1 rendering and metadata, including its original branch-dependent RNG consumption. It is intentionally not paired sample-for-sample with a new policy after the first differing palette branch. Regression tests retain image and metadata hashes captured before this extension for all three domain regimes.

New modes record generator version 2, `color_policy`, `train_color_correlation_by_experience`, `evaluation_color_correlation_by_experience`, and evaluator-only `train_color_change_flags`. Recorded `domain_parameters` describe the common independent evaluation condition; the separate training schedule supplies early overrides. Existing domain IDs and signal-change flags retain their original meaning and do not acquire a curriculum notification. `early_biased` also records the requested bias horizon and the number actually observed within the stream.

The new policies apply to recurring, stationary, and noise-only regimes. Sample counts and split RNG namespaces retain their existing behavior: changing training counts does not alter validation/test draws. A recurring domain repeats nuisance parameters with fresh samples.

These controls do not establish the absence of all shortcuts or demonstrate transfer to natural images. They support a controlled test of learning after early reliance on an intentionally misleading color feature. No v3 study or model tuning is part of this generator change.
