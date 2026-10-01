"""Schedule balance, causal selection and exact interrupted runner recovery."""

import copy
from collections import Counter
from pathlib import Path

import pytest
import torch

from acp_cl.acquisition.world import AcquisitionWorld, Law, batch
from acp_cl.predictive_value import study
from acp_cl.predictive_value.design import candidate_ids, counts, schedule, select, validate_config
from acp_cl.predictive_value.mechanism import state_signature
from acp_cl.replay_renewal.study import new_learner


def configuration():
    return dict(study.read_json('configs/predictive_value_smoke.json'),models=['conditional'],
                workers=1,cpu_affinity=0,checkpoint_every=16)


def test_schedule_balances_cue_mode_order_and_counts_all_exposure():
    c=study.read_json('configs/predictive_value_diagnostic.json')
    validate_config(c)
    assert counts(c)==dict(trajectories=24,assessments=48,
                           ordinary_arrivals_per_trajectory=20992,diagnostic_forks=1296)
    combinations=Counter()
    for seed in c['seeds']:
        phases=schedule(seed,c)
        scores=[p for p in phases if p['kind']=='score']
        assert sorted(p['gap'] for p in scores)==c['gaps']
        assert sum(p['size'] for p in phases)==20992
        combinations[scores[0]['cue'],seed%2,scores[0]['gap']]+=1
    assert len(combinations)==12 and set(combinations.values())=={1}


@pytest.mark.parametrize('key,value',[('candidate_count',15),('validation_size',32),
    ('gaps',[4096,512]),('batch_size',31),('cpu_affinity',-1),('interaction_features',False)])
def test_invalid_design_rejected(key,value):
    c=study.read_json('configs/predictive_value_diagnostic.json')
    c[key]=value
    with pytest.raises(ValueError):
        validate_config(c)


def test_candidate_sampling_is_distinct_reproducible_and_ignores_input_order():
    ids=list(range(16))
    selected,replacement=candidate_ids(ids,15,8,14001,0)
    assert len(set(selected+[replacement]))==9 and 15 not in selected+[replacement]
    assert (selected,replacement)==candidate_ids(ids[::-1],15,8,14001,0)
    assert ids==list(range(16))


def test_selection_uses_score_window_and_original_error_with_id_ties():
    candidates=[dict(id=7,original_brier=.2),dict(id=3,original_brier=.1),dict(id=5,original_brier=.1)]
    result=select(candidates,[.05,.08,.05,.2])
    assert result['value_id']==5 and result['accuracy_id']==3
    contaminated=copy.deepcopy(candidates)
    for row in contaminated:
        row['future_validation_loss']=-1000*row['id']
        row['origin_law']={'mode':1}
    assert select(contaminated,[.05,.08,.05,.2])==result
    assert study.observable_candidates(contaminated)==candidates


def test_locked_run_rejects_changed_configuration(tmp_path):
    c=configuration()
    protocol=Path('docs/predictive_value_protocol.md')
    result=study.run_suite(c,tmp_path,protocol=protocol,lock_only=True)
    assert result['locked']
    changed=dict(c,lr=.003)
    with pytest.raises(ValueError,match='resume requires exact'):
        study.run_suite(changed,tmp_path,resume=True,protocol=protocol,lock_only=True)


def test_checkpoint_roundtrip_preserves_parameter_gradients(tmp_path):
    torch.set_num_threads(1)
    c=configuration()
    learner=new_learner('conditional',14991,c)
    learner.train(batch(AcquisitionWorld(),Law(0),14991,'roundtrip',0,c['batch_size']))
    assert any(p.grad is not None for p in learner.model.parameters())
    path=tmp_path/'learner.pt'
    study.save(path,dict(identity={'test':1},learner=learner))
    restored=study.load(path,{'test':1})['learner']
    assert state_signature(restored)==state_signature(learner)


def test_interrupted_score_window_and_completed_resume_match_uninterrupted(tmp_path,monkeypatch):
    c=configuration()
    identity={'test':'interruption'}
    continuous=tmp_path/'continuous'
    resumed=tmp_path/'resumed'
    continuous.mkdir()
    resumed.mkdir()
    study.run_job(c,identity,14991,'conditional',continuous)
    original=study.save
    interrupted=False

    def fail_after_checkpoint(path,value):
        nonlocal interrupted
        original(path,value)
        if (not interrupted and Path(path).name=='checkpoint.pt' and Path(path).parent.name=='score_0'
                and value['record']['completed_packets']==2):
            interrupted=True
            raise RuntimeError('synthetic interruption after durable checkpoint')

    monkeypatch.setattr(study,'save',fail_after_checkpoint)
    with pytest.raises(RuntimeError,match='synthetic interruption'):
        study.run_job(c,identity,14991,'conditional',resumed)
    monkeypatch.setattr(study,'save',original)
    study.run_job(c,identity,14991,'conditional',resumed)
    left=study.load(continuous/'conditional_14991/return_1/checkpoint.pt',identity)['learner']
    right=study.load(resumed/'conditional_14991/return_1/checkpoint.pt',identity)['learner']
    assert state_signature(left,ignore_walltime=True)==state_signature(right,ignore_walltime=True)
    for index in (0,1):
        relative=f'conditional_14991/assessment_{index}/result.json'
        a,b=study.read_json(continuous/relative),study.read_json(resumed/relative)
        for key in ('choice','panels','candidates','replacement'):
            assert a[key]==b[key]
    original_result=(resumed/'conditional_14991/result.json').read_bytes()
    study.run_job(c,identity,14991,'conditional',resumed)
    assert (resumed/'conditional_14991/result.json').read_bytes()==original_result
    choice_path=resumed/'conditional_14991/assessment_0/choice.json'
    choice=study.read_json(choice_path)
    choice['choice']['value_id']=-999
    study.write_json(choice_path,choice)
    with pytest.raises(ValueError,match='selection differs'):
        study.run_job(c,identity,14991,'conditional',resumed)
