"""Prospective schedule and configuration for the core/residual comparison."""

from dataclasses import asdict

from acp_cl.acquisition.study import validate_config as validate_base
from acp_cl.acquisition.world import Law, order, stage_law


ARMS = ('joint', 'separate', 'fixed_features')
PHASES = ('maintenance', 'novel', 'return')
EXTRA = {'maintenance_size', 'return_size', 'residual_width',
         'residual_context_width', 'residual_decoder_width', 'cpu_affinity'}


def validate_config(config):
    validate_base({key:value for key,value in config.items() if key not in EXTRA})
    if not EXTRA <= config.keys() or config['kind'] != 'development':
        raise ValueError('fixed core/residual development configuration required')
    if any(type(config[key]) is not int or config[key] < 1 for key in EXTRA-{'cpu_affinity'}):
        raise ValueError('positive residual dimensions and phase sizes required')
    if type(config['cpu_affinity']) is not int or config['cpu_affinity'] < 0:
        raise ValueError('invalid affinity mask')
    if any(config[key] % config['probe_every'] for key in ('maintenance_size', 'return_size')):
        raise ValueError('phase sizes must be multiples of probe_every')
    if config['feedback_noise'] != 0:
        raise ValueError('this first architectural comparison uses clean feedback')
    if config['prefix_blocks'] % 2:
        raise ValueError('an even alternating prefix must end in the maintenance mode')
    if config['residual_width'] > config['width']:
        raise ValueError('the bounded residual cannot be wider than the core')


def prefix_phases(seed, config):
    return [dict(name=f'prefix_{block}', law=asdict(Law((seed+block) % 2)),
                 size=config['prefix_size'], cue=None, branch='novel', channel=f'prefix_{block}')
            for block in range(config['prefix_blocks'])]


def phases(seed, config, arm):
    if arm not in (*ARMS, 'fresh'):
        raise ValueError('unknown learning arm')
    result = [
        dict(name='maintenance', law=asdict(Law(1-seed % 2)), size=config['maintenance_size'],
             cue=None, branch='novel', channel='core_residual_maintenance'),
        dict(name='novel', law=asdict(stage_law(seed, 1)), size=config['episode_size'],
             cue=order(seed)[0], branch='novel', channel='core_residual_novel'),
        dict(name='return', law=asdict(Law(seed % 2)), size=config['return_size'],
             cue=None, branch='return', channel='core_residual_return')]
    return [result[1]] if arm == 'fresh' else result


def law_from_record(value):
    return Law(value['mode'], tuple(value['active']), value['revised'], value['noise'])


def counts(config):
    pairs = len(config['seeds'])*len(config['models'])
    arrivals = pairs*(config['prefix_blocks']*config['prefix_size']
        +len(ARMS)*(config['maintenance_size']+config['episode_size']+config['return_size'])
        +config['episode_size'])
    return dict(jobs=pairs, prefix_fits=pairs, prefix_phases=pairs*config['prefix_blocks'],
        learning_trajectories=pairs*len(ARMS), fresh_references=pairs,
        post_prefix_episodes=pairs*(len(ARMS)*len(PHASES)+1),
        total_phases=pairs*(config['prefix_blocks']+len(ARMS)*len(PHASES)+1),
        training_arrivals=arrivals, training_packets=arrivals//config['batch_size'],
        optimizer_steps=arrivals//config['batch_size']*config['updates_per_batch'])
