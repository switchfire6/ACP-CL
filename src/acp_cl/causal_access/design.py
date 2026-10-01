"""Fixed prospective design; no search or post-score setting selection."""

from dataclasses import asdict

from acp_cl.acquisition.study import validate_config as validate_base
from acp_cl.acquisition.world import Law, order, stage_law


EXTRA = {'maintenance_size', 'residual_width', 'residual_context_width',
         'residual_decoder_width', 'cpu_affinity', 'reference_size',
         'stream_blocks', 'stream_block_size', 'mixer_capacity', 'mixer_clip'}
POLICIES = ('core', 'full', 'half', 'causal', 'joint')


def validate_config(config):
    validate_base({key: value for key, value in config.items() if key not in EXTRA})
    if not EXTRA <= config.keys() or config['kind'] != 'development':
        raise ValueError('complete causal-access development configuration required')
    positive = EXTRA - {'cpu_affinity', 'mixer_clip'}
    if any(type(config[key]) is not int or config[key] < 1 for key in positive):
        raise ValueError('positive integer extension settings required')
    if config['cpu_affinity'] != 1365 or config['threads'] != 1:
        raise ValueError('use the declared six-core, single-thread resource setting')
    if any(config[key] % config['probe_every'] for key in ('maintenance_size', 'reference_size')):
        raise ValueError('training phases must be divisible by the checkpoint interval')
    if config['stream_block_size'] % config['batch_size'] or config['stream_blocks'] % 2:
        raise ValueError('even complete switching blocks required')
    if config['feedback_noise'] != 0 or config['mixer_capacity'] != 32 or config['mixer_clip'] != 1e-6:
        raise ValueError('this single experiment fixes feedback and mixer settings')
    if config['residual_width'] > config['width'] or config['evidence_strength'] != 1.:
        raise ValueError('keep the original bounded residual and evidence strength')
    if config['models'] != ['conditional', 'recurrent']:
        raise ValueError('both architectures must be included')
    main = dict(seeds=list(range(17001, 17007)), prefix_blocks=4, prefix_size=1024,
        maintenance_size=1024, episode_size=8192, reference_size=8192,
        batch_size=32, eval_size=512, probe_every=256, support_replicates=2,
        width=64, context_width=12, decoder_width=64, residual_width=32,
        residual_context_width=8, residual_decoder_width=32, memory_packets=16,
        updates_per_batch=12, workers=6, stream_blocks=16, stream_block_size=1024)
    smoke = dict(seeds=[17991], prefix_blocks=2, prefix_size=64,
        maintenance_size=32, episode_size=128, reference_size=128,
        batch_size=8, eval_size=64, probe_every=32, support_replicates=1,
        width=8, context_width=4, decoder_width=8, residual_width=4,
        residual_context_width=2, residual_decoder_width=4, memory_packets=4,
        updates_per_batch=1, workers=2, stream_blocks=4, stream_block_size=32)
    designs = {'causal_access_development': main, 'causal_access_smoke': smoke}
    expected = designs.get(config['study'])
    if expected is None or any(config[key] != value for key, value in expected.items()) or config['lr'] != .002:
        raise ValueError('configuration must match the fixed development or engineering smoke design')


def phases(seed, config):
    return [dict(name='maintenance', law=asdict(Law(1-seed % 2)),
                 size=config['maintenance_size'], cue=None, branch='novel',
                 channel='core_residual_maintenance'),
            dict(name='novel', law=asdict(stage_law(seed, 1)),
                 size=config['episode_size'], cue=order(seed)[0], branch='novel',
                 channel='core_residual_novel')]


def reference_phase(seed, config, condition):
    law = Law(seed % 2) if condition == 'old' else stage_law(seed, 1)
    return dict(name=f'reference_{condition}', law=asdict(law), size=config['reference_size'],
                cue=None if condition == 'old' else order(seed)[0], branch='novel',
                channel=f'causal_access_reference_{condition}')


def counts(config):
    jobs = len(config['seeds']) * len(config['models'])
    arrivals = jobs * (config['prefix_blocks'] * config['prefix_size']
        + 2 * (config['maintenance_size'] + config['episode_size'])
        + 2 * config['reference_size'])
    return dict(jobs=jobs, total_phases=jobs * (config['prefix_blocks'] + 6),
                training_arrivals=arrivals, training_packets=arrivals // config['batch_size'],
                optimizer_steps=arrivals // config['batch_size'] * config['updates_per_batch'],
                stream_arrivals=jobs * config['stream_blocks'] * config['stream_block_size'])
