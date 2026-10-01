"""Prospective state-substitution design with paired environmental branches."""

from dataclasses import asdict

import numpy as np

from acp_cl.acquisition.world import AcquisitionWorld, Law, order, stage_law
from acp_cl.persistence.study import trial_seed
from acp_cl.predictive_value.design import validate_config as validate_previous


CELLS = tuple(f'{i:03b}' for i in range(8))
BRANCHES = ('current', 'return')


def validate_config(config):
    if 'inter_assessment_size' not in config or 'gaps' in config or 'recovery_size' in config:
        raise ValueError('state diagnostic requires its explicit inter-assessment exposure')
    size = config['inter_assessment_size']
    if type(size) is not int or size <= 0 or size % config['batch_size']:
        raise ValueError('positive packet-aligned inter-assessment exposure required')
    previous = {key:value for key,value in config.items() if key != 'inter_assessment_size'}
    previous.update(gaps=[config['batch_size'], 2*config['batch_size']],
                    recovery_size=config['batch_size'])
    validate_previous(previous)


def schedule(seed, config):
    phases = []

    def add(name, law, size, kind='train', assessment=None):
        law = asdict(law)
        law['active'] = list(law['active'])
        phases.append(dict(name=name, law=law, size=size, kind=kind,
                           assessment=assessment, gap=None, cue=order(seed)[0]))

    for index in range(config['prefix_blocks']):
        add(f'prefix_{index}', Law((seed+index) % 2), config['prefix_size'])
    current = stage_law(seed, 1)
    add('acquisition', current, config['acquisition_size'])
    for index in range(2):
        add(f'score_{index}', current, config['score_size'], 'score', index)
        add(f'validation_{index}', current, config['validation_size'], 'train', index)
        if index == 0:
            add('inter_assessment', current, config['inter_assessment_size'])
    return phases


def counts(config):
    trajectories = len(config['seeds'])*len(config['models'])
    assessments = 2*trajectories
    candidates = config['candidate_count']+1
    score_packets = config['score_size']//config['batch_size']
    validation_packets = config['validation_size']//config['batch_size']
    return dict(trajectories=trajectories, assessments=assessments,
        ordinary_arrivals_per_trajectory=sum(p['size'] for p in schedule(config['seeds'][0], config)),
        diagnostic_forks=assessments*len(CELLS)*candidates,
        instrumented_forks=assessments*(len(CELLS)-1)*candidates,
        zero_update_copies=2*assessments,
        diagnostic_prediction_forwards=assessments*(score_packets*candidates
            +len(BRANCHES)*validation_packets*(len(CELLS)*candidates+2)),
        extra_loss_forwards=4*assessments*(len(CELLS)-1)*candidates)


def paired_batch(seed, assessment, index, config):
    """Change the law with the same exogenous draw and performed actions."""
    world = AcquisitionWorld()
    draw = trial_seed(seed, f'acquisition_predictive_value_validation_{assessment}', index)
    laws = (stage_law(seed, 1), Law(seed % 2))
    experiences, truths = [], []
    for law in laws:
        data = world.experience(law, config['batch_size'], draw)
        cases = world.dataset(law, config['batch_size'], draw)
        truth = world.counterfactuals(cases)
        if (not np.array_equal(cases.observations, data.observations)
                or not np.array_equal(truth[np.arange(len(data)), data.actions], data.survival)):
            raise AssertionError('clean performed feedback differs from evaluator physics')
        experiences.append(data)
        truths.append(truth)
    if (not np.array_equal(experiences[0].observations, experiences[1].observations)
            or not np.array_equal(experiences[0].actions, experiences[1].actions)):
        raise AssertionError('paired environmental branches differ in observations/actions')
    return experiences, truths
