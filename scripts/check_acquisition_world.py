"""Check genuine physical changes, previously irrelevant cues, and conservation."""

import argparse
from dataclasses import replace
import json

import numpy as np

from acp_cl.acquisition.world import AcquisitionWorld, Law, ORDERS, mask_valid, signal_images
from acp_cl.persistence.study import write_json


def check(size=4096):
    world, rows, residual = AcquisitionWorld(), [], 0.
    for mode in (0, 1):
        root = world.dataset(Law(mode), size, 381077)
        original = world.counterfactuals(root)
        for ordering in ORDERS:
            for stage, cue in enumerate(ordering, 1):
                before = replace(root, law=Law(mode, tuple(sorted(ordering[:stage-1]))))
                after = replace(root, law=Law(mode, tuple(sorted(ordering[:stage]))))
                signals = root.signals.copy()
                signals[:, cue] ^= 1
                changed = dict(signals=signals, observations=signal_images(root.observations, signals))
                old_a, old_b = (world.counterfactuals(c) for c in (before, replace(before, **changed)))
                np.testing.assert_array_equal(old_a, old_b)
                a, b = (world.counterfactuals(c) for c in (after, replace(after, **changed)))
                valid = mask_valid(after)
                np.testing.assert_array_equal(a[valid], original[valid])
                # A cue-dependent action may outperform any single action shared
                # by this matched pair. This is a clairvoyant feasibility bound.
                terminal_a, terminal_b = a[:, :, -1].astype(float), b[:, :, -1].astype(float)
                utility = ((terminal_a.max(axis=1)+terminal_b.max(axis=1))/2
                           - ((terminal_a+terminal_b)/2).max(axis=1)).mean()
                rows.append(dict(mode=mode, order=list(ordering), stage=stage, cue=cue,
                    changed_action_outcomes=float((a != b).any(axis=(1, 2)).mean()),
                    clairvoyant_cue_value=float(utility),
                    action_opportunity=float(terminal_a.max(axis=1).mean()-terminal_a[:, 0].mean())))
                if not np.any(a != b):
                    raise AssertionError("new cue has no predictive consequence")
                for action in range(5):
                    result = world.simulate(after, np.full(size, action))
                    residual = max(residual, result.conservation_error)
                    if result.min_stock < 0 or result.conservation_error > 1e-10:
                        raise AssertionError("physical conservation failed")
        final = replace(root, law=Law(mode, (0, 1, 2)))
        revised = replace(final, law=replace(final.law, revised=True))
        valid = mask_valid(final, "revision")
        a, b = world.counterfactuals(final), world.counterfactuals(revised)
        np.testing.assert_array_equal(a[valid], b[valid])
        if not np.any(a[~valid] != b[~valid]):
            raise AssertionError("revision did not invalidate any previous consequence")
        noisy = replace(final, law=replace(final.law, noise=.2))
        np.testing.assert_array_equal(world.counterfactuals(final), world.counterfactuals(noisy))
    return dict(passed=True, size_per_panel=size, largest_conservation_error=residual, rows=rows,
        scope="Structural checks and clairvoyant action feasibility, not learned acquisition adequacy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check()
    write_json(args.output, result)
    print(json.dumps(dict(passed=result["passed"], largest_conservation_error=result["largest_conservation_error"],
        min_cue_value=min(r["clairvoyant_cue_value"] for r in result["rows"]),
        min_action_opportunity=min(r["action_opportunity"] for r in result["rows"])), indent=2))
