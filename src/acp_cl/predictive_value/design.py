"""Fixed schedule, candidate sampling and causal value selection."""

from dataclasses import asdict

import numpy as np

from acp_cl.acquisition.world import Law, order, stage_law
from acp_cl.persistence.study import trial_seed


def validate_config(c):
    fields = {'study', 'kind', 'seeds', 'models', 'prefix_blocks', 'prefix_size',
              'acquisition_size', 'recovery_size', 'score_size', 'validation_size',
              'gaps', 'batch_size', 'checkpoint_every', 'candidate_count', 'rehearsal_updates',
              'width', 'experts', 'context_width', 'decoder_width', 'interaction_features',
              'lr', 'memory_packets', 'updates_per_batch', 'evidence_strength',
              'threads', 'workers', 'cpu_affinity'}
    if set(c) != fields or c['kind'] != 'diagnostic':
        raise ValueError('invalid predictive-value configuration')
    nonints = {'study', 'kind', 'seeds', 'models', 'gaps', 'interaction_features',
               'lr', 'evidence_strength', 'cpu_affinity'}
    if any(type(c[k]) is not int or c[k] < 1 for k in fields-nonints):
        raise ValueError('positive integer configuration required')
    if type(c['cpu_affinity']) is not int or c['cpu_affinity'] < 0:
        raise ValueError('invalid affinity')
    if c['prefix_blocks'] < 2 or c['prefix_blocks'] % 2 or c['experts'] != 4:
        raise ValueError('even prefix count and four hypotheses required')
    if c['batch_size'] % 8 or c['memory_packets'] < c['candidate_count']+2:
        raise ValueError('invalid batch or bounded candidate capacity')
    if len(c['gaps']) != 2 or any(type(g) is not int or g < 1 for g in c['gaps']) or c['gaps'][0] >= c['gaps'][1]:
        raise ValueError('two increasing positive gaps required')
    sizes = [c[k] for k in ('prefix_size', 'acquisition_size', 'recovery_size',
                            'score_size', 'validation_size', 'checkpoint_every')]+c['gaps']
    if any(n % c['batch_size'] for n in sizes):
        raise ValueError('exposure must be a multiple of packet size')
    if c['validation_size'] < 2*c['batch_size']:
        raise ValueError('validation requires initial and later packets')
    if c['interaction_features'] is not True:
        raise ValueError('qualified recurrent interactions required')
    if any(not np.isfinite(c[k]) or c[k] <= 0 for k in ('lr', 'evidence_strength')):
        raise ValueError('invalid continuous setting')
    if not c['seeds'] or len(set(c['seeds'])) != len(c['seeds']) or any(type(s) is not int or s < 0 for s in c['seeds']):
        raise ValueError('distinct nonnegative integer seeds required')
    if not c['models'] or len(set(c['models'])) != len(c['models']) or any(m not in ('conditional', 'recurrent') for m in c['models']):
        raise ValueError('invalid models')


def schedule(seed, c):
    phases = []

    def add(name, law, size, kind='train', assessment=None, gap=None):
        description = asdict(law)
        description['active'] = list(description['active'])
        phases.append(dict(name=name, law=description, size=size, kind=kind,
                           assessment=assessment, gap=gap, cue=order(seed)[0]))

    for index in range(c['prefix_blocks']):
        add(f'prefix_{index}', Law((seed+index) % 2), c['prefix_size'])
    current = stage_law(seed, 1)
    add('acquisition', current, c['acquisition_size'])
    gaps = c['gaps'][::-1] if (seed//6) % 2 else c['gaps']
    for index, gap in enumerate(gaps):
        add(f'score_{index}', current, c['score_size'], 'score', index, gap)
        add(f'near_{index}', current, c['validation_size'], 'near', index, gap)
        add(f'gap_{index}', current, gap, 'train', index, gap)
        add(f'return_{index}', Law(seed % 2), c['validation_size'], 'return', index, gap)
        if index == 0:
            add('recovery', current, c['recovery_size'])
    return phases


def law_from_record(record):
    return Law(record['mode'], tuple(record['active']), record['revised'], record['noise'])


def candidate_ids(memory_ids, latest_id, count, seed, assessment):
    pool = sorted(set(memory_ids)-{latest_id})
    if len(pool) < count+1:
        raise ValueError('insufficient distinct non-anchor packets')
    rng = np.random.default_rng(trial_seed(seed, 'predictive_value_candidates', assessment))
    chosen = rng.choice(pool, count+1, replace=False).tolist()
    return sorted(chosen[:count]), int(chosen[-1])


def select(candidates, score_losses):
    ids = [row['id'] for row in candidates]
    losses = np.asarray(score_losses, dtype=np.float64)
    errors = np.asarray([row['original_brier'] for row in candidates], dtype=np.float64)
    if len(set(ids)) != len(ids) or losses.shape != (len(ids)+1,) or not np.isfinite(losses).all() or not np.isfinite(errors).all():
        raise ValueError('invalid score selection inputs')
    values = losses[-1]-losses[:-1]
    best = min(range(len(ids)), key=lambda i: (-values[i], ids[i]))
    accurate = min(range(len(ids)), key=lambda i: (errors[i], ids[i]))
    return dict(value_id=ids[best], accuracy_id=ids[accurate],
                score_losses=losses.tolist(), values=values.tolist())


def counts(c):
    trajectories = len(c['seeds'])*len(c['models'])
    arrivals = sum(p['size'] for p in schedule(c['seeds'][0], c))
    return dict(trajectories=trajectories, assessments=2*trajectories,
                ordinary_arrivals_per_trajectory=arrivals,
                diagnostic_forks=6*(c['candidate_count']+1)*trajectories)
