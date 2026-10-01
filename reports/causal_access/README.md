# Causal access to frozen predictive functions

Protocol: [causal_access_protocol.md](../../docs/causal_access_protocol.md).
Implementation: `src/acp_cl/causal_access`; independent scientific scoring:
`scripts/summarize_causal_access.py`.

The development cohort uses six fresh seeds and both architectures, followed
by 16 unannounced alternating old/novel blocks. Every transition contributes
to the scores. Specialist and component qualification is separate from the
relative policy comparisons. Passing requires qualification and all fixed
performance, cue-use, survival and attribution requirements.

- `development/summary.json`: raw per-seed metrics and every fixed gate.
- `development/summary.md`: compact independent numerical report.
- `development/*.png` and `*.svg`: presentation-only views of saved scores.
- `smoke_v2/summary.json`: successful engineering reconstruction; ineligible.
- `../causal_access_checks/`: implementation, forecast and preservation checks.
- `../../runs/causal_access_development/`: source/protocol/analysis locks,
  completed fits, original forecasts and all raw scoring inputs.
- `../../runs/causal_access_smoke/`: preserved first smoke with the pre-main
  count-dtype validation mismatch; see the engineering record.

This test permits imperfect retention through fixed error and survival
tolerances. It tests access to existing predictive functions; it does not
measure consolidation, an optimal forgetting rate or continuous representation
learning after the frozen stream begins.
