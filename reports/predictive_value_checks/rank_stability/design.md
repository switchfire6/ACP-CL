# Retrospective diagnosis of predictive-value rank stability

This design is specified **after seeing the completed primary failures and
secondary summaries** of the predictive-value experiment. It is exploratory,
not a prospective experiment, confirmation, new gate or permission to retune.
It uses existing raw assessment records only: no fitting, new observations,
memory-policy change or modification of the original decisions.

The input is `reports/predictive_value/diagnostic/raw_results.jsonl.gz`, containing
48 assessments from 12 seeds and two architectures. Its compressed-file SHA256,
this design and the isolated analysis script are locked before computing these
correlations. Archive the script/design in `source.zip` and their hashes in
`source_lock.json`. Existing training, analysis, protocol and report files remain
unchanged.

For each assessment, align the eight candidates in their recorded order and
exclude the ninth, common replacement. Use candidate Brier losses from the score
window and the four complete later panels. Compute six tied-average Spearman
correlations:

1. Score versus near frozen.
2. Score versus near reapplied.
3. Score versus return frozen.
4. Score versus return reapplied.
5. Near frozen versus near reapplied.
6. Return frozen versus return reapplied.

Assign ascending ranks starting at one. An exact tie occupying ordered positions
a through b receives rank (a+b)/2. For rank vectors r and q, compute

    rho = sum_i[(r_i-r_bar)(q_i-q_bar)]
          / sqrt(sum_i(r_i-r_bar)^2 * sum_i(q_i-q_bar)^2).

Both vectors are losses, so a positive correlation means their ordering agrees.
Ties are defined by exact recorded numerical equality, with no fitted tolerance.
If either vector is constant, report an undefined correlation and identify the
constant side. Do not silently omit it. A seed mean requires both assessments
to be defined; the overall mean requires all 12 seed means. Report undefined
counts and their identities even if this prevents a pooled mean.

Average the two assessment correlations within each seed, then average those
12 seed means separately by architecture. Report every seed value and the number
of positive, zero, negative and undefined seed means. Do not pool candidates,
horizons or assessments as independent observations. No inferential intervals,
p-values or new pass/fail thresholds are calculated.

Before reading the cohort vectors, verify known perfect, opposite, tied and
constant examples. The tied example x=(1,1,2,3), y=(4,5,5,6) has rho=5/6.
Retain individual assessment correlations, tie counts and candidate IDs in JSON.

This analysis asks whether ordering survives later queries and reapplication.
Frozen and reapplied panels share evaluation queries and context, but differ in
parent parameters, Adam state and latest anchor. Neither their comparison nor
the two gap lengths isolate a causal age effect. Rank agreement does not measure
gain size, calibration, survival, long-term retention or policy effectiveness.
Small candidate differences can be noisy; choosing the minimum score-window
loss is optimistic on that same window. Common-replacement subtraction leaves
candidate rankings unchanged. Source-law coverage and temporal dependence remain
limitations; no subgroup is dropped to improve the result.

Any pattern here is a retrospective explanation to check independently later.
The failed local and return criteria of the original experiment remain failed.
