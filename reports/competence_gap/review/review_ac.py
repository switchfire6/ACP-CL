"""Reviewer checks A (independent recompute from raw arrays) and C (leakage/confounds).

Uses only raw run artifacts plus the project simulator; does not import scripts/competence_gap.py.
"""
import sys
import json
import hashlib
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from acp_cl.acquisition.world import AcquisitionWorld, stage_law
from acp_cl.persistence.study import trial_seed
from acp_cl.persistence.world import Experience

RUN = ROOT / 'runs' / 'competence_gap'
SEEDS = list(range(19101, 19107))
FACT = [f'U{u}_R{r}_W{w}' for u in (12, 3) for r in (16, 256) for w in (64, 256)]
ARMS = FACT + ['exposure', 'all_action']
REF = 'U12_R16_W64'
world = AcquisitionWorld()
out = dict(A={}, C={})


def brier(p, t):
    return float(((p.astype(np.float64) - t.astype(np.float64)) ** 2).mean())


def img_hashes(obs):
    return {hashlib.sha256(o.tobytes()).hexdigest() for o in obs}


O, O_raw, O_gauge, EX, BR = {}, {}, {}, {}, {}
monotone_learner_violations = 0
checks = dict(panel_truth_regenerated=True, support_regenerated=True, correct_obs_equal_panel=True,
              fits_not_resumed=True)
C = dict(stream_fingerprints_regenerated=True, training_truth_regenerated=True,
         panel_train_image_overlap=0, support_train_image_overlap=0, init_by_width={}, memory={},
         all_action_vs_ref={}, action_counts={}, action_chi2={}, exposure_prefix={})
for seed in SEEDS:
    law = stage_law(seed, 3)
    cases = world.dataset(law, 512, trial_seed(seed, 'acquisition_query', 0))
    truth = world.counterfactuals(cases)
    supports = [world.experience(law, 32, trial_seed(seed, 'acquisition_support', [r, False])) for r in range(2)]
    with np.load(RUN / 'oracle' / f'seed_{seed}.npz') as z:
        oz = {k: z[k] for k in z.files}
    checks['panel_truth_regenerated'] &= bool(np.array_equal(oz['panel_truth'], truth))
    p = oz['endpoint_exact'][0]
    O_raw[seed] = brier(p, truth)
    O[seed] = brier(np.minimum.accumulate(p, axis=-1), truth)
    O_gauge[seed] = brier(np.minimum.accumulate(oz['endpoint_gauge'][0], axis=-1), truth)
    # ---- regenerate the training stream (32768 arrivals)
    fps, obs_hashes, tr_truth, tr_actions, tr_out = [], set(), [], [], []
    for i in range(1024):
        ds = trial_seed(seed, 'competence_gap_training', i)
        d = world.experience(law, 32, ds)
        c = world.dataset(law, 32, ds)
        fps.append(d.fingerprint())
        obs_hashes |= img_hashes(d.observations)
        tr_truth.append(world.counterfactuals(c))
        tr_actions.append(d.actions)
        tr_out.append(d.survival)
    tr_truth, tr_actions, tr_out = map(np.concatenate, (tr_truth, tr_actions, tr_out))
    C['panel_train_image_overlap'] += len(img_hashes(cases.observations) & obs_hashes)
    C['support_train_image_overlap'] += sum(len(img_hashes(s.observations) & obs_hashes) for s in supports)
    counts = np.bincount(tr_actions[:8192], minlength=5)
    C['action_counts'][str(seed)] = counts.tolist()
    C['action_chi2'][str(seed)] = float(((counts - 8192 / 5) ** 2 / (8192 / 5)).sum())
    init = {}
    for arm in ARMS:
        d = RUN / 'fits' / f'{arm}_{seed}'
        rec = json.loads((d / 'result.json').read_text())
        checks['fits_not_resumed'] &= (not rec['resumed']) and not rec['resumes']
        n = len(rec['fingerprints'])
        C['stream_fingerprints_regenerated'] &= rec['fingerprints'] == fps[:n]
        with np.load(d / 'training.npz') as t:
            tt = {k: t[k] for k in t.files}
        N = len(tt['truth'])
        C['training_truth_regenerated'] &= bool(np.array_equal(tt['truth'], tr_truth[:N])
                                                and np.array_equal(tt['actions'], tr_actions[:N])
                                                and np.array_equal(tt['outcomes'], tr_out[:N]))
        init.setdefault(rec['spec']['W'], set()).add(rec['initial_model_sha256'])
        C['memory'][f'{arm}_{seed}'] = dict(
            R=rec['spec']['R'], packets_at_scores=[s['memory_packets'] for s in rec['scores']],
            capacity=[s['memory_capacity'] for s in rec['scores']],
            ids_min_max=[min(rec['diagnostics']['memory_ids']), max(rec['diagnostics']['memory_ids'])],
            n_ids=len(rec['diagnostics']['memory_ids']), optimizer_steps=rec['optimizer_steps'], U=rec['spec']['U'])
        recs = json.loads((d / 'evaluations.json').read_text())['records']
        with np.load(d / 'evaluations.npz') as z:
            for point in range(len(recs) // 8):
                briers = []
                for r in recs[point * 8:(point + 1) * 8]:
                    if r['cue'] is None and not r['flipped']:
                        k = r['index']
                        pr, tru, ob = z[f'e{k}_probabilities'], z[f'e{k}_truth'], z[f'e{k}_observations']
                        checks['panel_truth_regenerated'] &= bool(np.array_equal(tru, truth))
                        checks['correct_obs_equal_panel'] &= bool(np.array_equal(ob, cases.observations))
                        sfp = Experience(z[f'e{k}_support_observations'], z[f'e{k}_support_actions'],
                                         z[f'e{k}_support_outcomes']).fingerprint()
                        checks['support_regenerated'] &= sfp in {s.fingerprint() for s in supports}
                        monotone_learner_violations += int((np.diff(pr, axis=-1) > 0).sum())
                        briers.append(brier(pr, tru))
                assert len(briers) == 2
                arrivals = rec['scores'][point]['arrivals']
                BR[seed, arm, arrivals] = float(np.mean(briers))
                EX[seed, arm, arrivals] = BR[seed, arm, arrivals] - O[seed]
    C['init_by_width'][str(seed)] = {str(k): len(v) for k, v in init.items()}
    ref = json.loads((RUN / 'fits' / f'{REF}_{seed}' / 'result.json').read_text())
    aa = json.loads((RUN / 'fits' / f'all_action_{seed}' / 'result.json').read_text())
    C['all_action_vs_ref'][str(seed)] = dict(
        same_init=ref['initial_model_sha256'] == aa['initial_model_sha256'],
        same_memory_ids=ref['diagnostics']['memory_ids'] == aa['diagnostics']['memory_ids'],
        same_optimizer_steps=ref['optimizer_steps'] == aa['optimizer_steps'],
        same_replay_presentations=(ref['diagnostics']['cost']['replay_presentations']
                                   == aa['diagnostics']['cost']['replay_presentations']),
        same_settings=ref['settings'] == aa['settings'], learner_class=aa['learner_class'],
        privileged_label_presentations=aa['diagnostics'].get('privileged_label_presentations'))
    with np.load(RUN / 'fits' / f'{REF}_{seed}' / 'training.npz') as a, \
            np.load(RUN / 'fits' / f'exposure_{seed}' / 'training.npz') as b:
        C['exposure_prefix'][str(seed)] = bool(all(np.array_equal(a[k], b[k][:len(a[k])]) for k in a.files))
    print(seed, 'done', flush=True)


def ex(arm, arrivals=8192):
    return np.array([EX[s, arm, arrivals] for s in SEEDS])


ref = ex(REF)
refm = ref.mean()
levels = dict(U=(12, 3), R=(16, 256), W=(64, 256))
effects = {}
for f, (lo, hi) in levels.items():
    others = [k for k in levels if k != f]
    diffs = []
    for x in levels[others[0]]:
        for y in levels[others[1]]:
            v = {others[0]: x, others[1]: y}
            v[f] = hi
            a = ex(f"U{v['U']}_R{v['R']}_W{v['W']}")
            v[f] = lo
            b = ex(f"U{v['U']}_R{v['R']}_W{v['W']}")
            diffs.append(a - b)
    e = np.mean(diffs, axis=0)
    effects[f] = dict(mean=float(e.mean()), fraction=float(-e.mean() / refm), reduce=int((e < 0).sum()),
                      increase=int((e > 0).sum()), per_seed=e.tolist())
res = json.loads((ROOT / 'reports/competence_gap/results.json').read_text())
arm_means = {a: float(ex(a).mean()) for a in ARMS}
arm_means['exposure@final'] = float(ex('exposure', 32768).mean())
closed_aa = ref - ex('all_action')
maxdiff = 0.
for a in ARMS:
    key = 'exposure@arrivals' if a == 'exposure' else a
    for s in SEEDS:
        maxdiff = max(maxdiff, abs(res['arms'][key]['excess']['per_seed'][str(s)] - EX[s, a, 8192]))
for s in SEEDS:
    maxdiff = max(maxdiff, abs(res['arms']['exposure@final']['excess']['per_seed'][str(s)] - EX[s, 'exposure', 32768]))
out['A'] = dict(
    O_brier_per_seed={str(s): O[s] for s in SEEDS}, O_brier_mean=float(np.mean(list(O.values()))),
    O_cummin_changes_brier=float(max(abs(O[s] - O_raw[s]) for s in SEEDS)),
    O_gauge_brier_mean=float(np.mean(list(O_gauge.values()))),
    learner_monotonicity_violations=monotone_learner_violations,
    reference_excess_mean=float(refm), reference_excess_per_seed=ref.tolist(),
    arm_excess_mean=arm_means, effects=effects,
    all_action=dict(excess=float(ex('all_action').mean()), fraction=float(closed_aa.mean() / refm),
                    seeds_improved=int((closed_aa > 0).sum())),
    exposure=dict(at8192=float(ex('exposure').mean()), at32768=arm_means['exposure@final'],
                  equal_ref_at8192=bool(np.array_equal(ex('exposure'), ref)),
                  seeds_reduced=int((ref - ex('exposure', 32768) > 0).sum())),
    best_candidate=min((k for k in arm_means if k != 'all_action'), key=arm_means.get),
    any_arm_le_010=[k for k, v in arm_means.items() if v <= .010],
    max_abs_diff_vs_results_json_per_seed_excess=maxdiff,
    results_json_effects={k: dict(mean=v['effect_other_minus_reference']['mean'],
                                  fraction=v['fraction_of_reference_excess_removed'],
                                  reduce=v['seeds_reducing_excess']) for k, v in res['main_effects'].items()},
    per_seed_brier={f'{a}_{s}_{n}': BR[s, a, n] for (s, a, n) in BR}, checks=checks)
C['init_shared_per_width'] = all(all(v == 1 for v in d.values()) for d in C['init_by_width'].values())
out['C'] = C
(ROOT / 'reports/competence_gap/review/review_ac.json').write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out['A'].items() if k != 'per_seed_brier'}, indent=1))
print(json.dumps({k: v for k, v in C.items() if k != 'memory'}, indent=1))
