"""Prospective schedules for a bounded temporal-representation experiment."""

from dataclasses import asdict

from acp_cl.acquisition.world import Law, order, stage_law


ARMS = ('outcome', 'final_frame', 'sequence', 'compute')


def law_from_record(value):
    return Law(value['mode'], tuple(value['active']), value['revised'], value['noise'])


def law_key(law):
    return f"m{law.mode}_a{''.join(str(i) for i in law.active)}_r{int(law.revised)}"


def reference_laws(seed):
    return [Law(seed % 2), Law(1-seed % 2), *[stage_law(seed, s) for s in (1, 2, 3)]]


def block(law, size):
    return dict(law=asdict(law), size=size)


def qualification_phases(seed, config):
    result = [dict(name=f'fresh_{stage}', cue=order(seed)[stage-1], branch='novel',
                   blocks=[block(stage_law(seed, stage), config['episode_size'])],
                   targets=[asdict(stage_law(seed, stage))]) for stage in (1, 2, 3)]
    laws = reference_laws(seed)
    blocks = [block(law, config['interleave_block'])
              for _ in range(config['interleave_cycles']) for law in laws]
    result.append(dict(name='interleaved', cue=None, branch='novel', blocks=blocks,
                       targets=[asdict(law) for law in laws]))
    return result


def main_phases(seed, config):
    result = [dict(name=f'prefix_{i}', cue=None, branch='novel',
                   blocks=[block(Law((seed+i) % 2), config['prefix_size'])], targets=[])
              for i in range(4)]
    for stage in (1, 2, 3):
        law = stage_law(seed, stage)
        result.append(dict(name=f'novel_{stage}', cue=order(seed)[stage-1], branch='novel',
                           blocks=[block(law, config['episode_size'])], targets=[asdict(law)]))
    newest = stage_law(seed, 3)
    revised = Law(newest.mode, newest.active, revised=True)
    for name, law, cue, branch in (
            ('return_old', Law(seed % 2), None, 'return'),
            ('return_new', newest, None, 'novel'),
            ('revision', revised, 0, 'revision'),
            ('return_unrevised', newest, 0, 'novel')):
        result.append(dict(name=name, cue=cue, branch=branch,
                           blocks=[block(law, config['episode_size'])], targets=[asdict(law)]))
    return result


def phase_laws(phase, batch_size):
    return [law_from_record(item['law']) for item in phase['blocks']
            for _ in range(item['size']//batch_size)]


def validate_config(c):
    expected = {'study', 'kind', 'seeds', 'arms', 'batch_size', 'eval_size',
                'probe_every', 'support_replicates', 'width', 'experts', 'context_width',
                'decoder_width', 'interaction_features', 'lr', 'memory_packets',
                'updates_per_batch', 'compute_updates_per_batch', 'auxiliary_weight',
                'evidence_strength', 'threads', 'workers', 'cpu_affinity', 'episode_size',
                'prefix_size', 'interleave_block', 'interleave_cycles', 'smoke'}
    if set(c) != expected:
        raise ValueError(f'invalid configuration fields: {set(c) ^ expected}')
    if c['kind'] not in ('qualification', 'main') or type(c['smoke']) is not bool:
        raise ValueError('invalid study kind')
    if (not c['seeds'] or len(set(c['seeds'])) != len(c['seeds'])
            or any(type(s) is not int or s < 0 for s in c['seeds'])):
        raise ValueError('unique nonnegative seeds required')
    if c['arms'] != (['outcome'] if c['kind'] == 'qualification' else list(ARMS)):
        raise ValueError('all declared arms required')
    integers = expected - {'study', 'kind', 'seeds', 'arms', 'interaction_features',
                           'lr', 'auxiliary_weight', 'evidence_strength', 'smoke'}
    if any(type(c[k]) is not int or c[k] < 1 for k in integers):
        raise ValueError('positive integer settings required')
    if c['batch_size'] % 8 or c['eval_size'] % 64:
        raise ValueError('factorial batch/evaluation sizes required')
    if any(c[k] % c['batch_size'] for k in ('episode_size', 'prefix_size',
                                         'interleave_block', 'probe_every')):
        raise ValueError('whole training batches required')
    if c['episode_size'] != c['interleave_cycles'] * c['interleave_block']:
        raise ValueError('each interleaved law must get one reference exposure budget')
    if not c['interaction_features'] or c['experts'] != 4:
        raise ValueError('keep the reference recurrent backbone')
    if (c['lr'] != .002 or c['auxiliary_weight'] != 1. or c['evidence_strength'] != 1.
            or c['threads'] != 1 or c['cpu_affinity'] != 1365):
        raise ValueError('fixed numerical and resource settings required')
    if c['compute_updates_per_batch'] < c['updates_per_batch']:
        raise ValueError('compute control must receive at least the base updates')
    if not c['smoke']:
        fixed = dict(batch_size=32, eval_size=512, support_replicates=2,
                     width=64, context_width=12, decoder_width=64, memory_packets=16,
                     updates_per_batch=12, episode_size=8192, prefix_size=1024,
                     interleave_block=1024, interleave_cycles=8, probe_every=1024, workers=6)
        if any(c[k] != v for k, v in fixed.items()):
            raise ValueError('scientific run must use the fixed prospective recipe')
        seeds = list(range(18101, 18107) if c['kind'] == 'qualification'
                     else range(18201, 18207))
        if c['seeds'] != seeds:
            raise ValueError('use the declared disjoint counterbalanced seed cohort')


def counts(config):
    phase_fn = qualification_phases if config['kind'] == 'qualification' else main_phases
    per_arm = sum(sum(b['size'] for b in p['blocks'])
                  for p in phase_fn(config['seeds'][0], config))
    jobs = len(config['seeds']) * len(config['arms'])
    steps = sum((config['compute_updates_per_batch'] if arm == 'compute'
                 else config['updates_per_batch']) * per_arm//config['batch_size']
                for arm in config['arms']) * len(config['seeds'])
    return dict(jobs=jobs, phases=jobs*len(phase_fn(config['seeds'][0], config)),
                training_arrivals=jobs*per_arm, optimizer_steps=steps)
