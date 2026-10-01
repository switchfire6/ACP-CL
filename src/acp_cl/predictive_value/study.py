"""Atomic, source-locked prospective rehearsal-value diagnostic."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import torch

from acp_cl.acquisition.study import load as load_checkpoint
from acp_cl.acquisition.world import AcquisitionWorld, batch
from acp_cl.conditional.learner import Packet
from acp_cl.persistence.learner import state_hash
from acp_cl.persistence.study import digest, trial_seed
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import affinity, atomic_checkpoint, new_learner, write_json
from acp_cl.selective_updates.study import source_manifest as prior_sources, verify_analysis
from .design import candidate_ids, counts, law_from_record, schedule, select, validate_config
from .mechanism import make_shadow, packet_fingerprint, predict_all, state_signature


ROOT = Path(__file__).resolve().parents[3]


def source_manifest():
    files = prior_sources()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        files[path.relative_to(ROOT/'src').as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def array_hash(values):
    values = np.asarray(values)
    return hashlib.sha256(str((values.shape, values.dtype.str)).encode()+values.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def observable_candidates(metadata):
    return [dict(id=row['id'],original_brier=row['original_brier']) for row in metadata]


def save(path, payload):
    # Parameter.__reduce__ does not preserve .grad. Keep exact state explicitly.
    value = dict(payload)
    learners = ([value['learner']] if 'learner' in value else value.get('learners', []))
    value['saved_gradients'] = [{name:None if p.grad is None else p.grad.detach().cpu().clone()
                                 for name,p in learner.model.named_parameters()} for learner in learners]
    atomic_checkpoint(path,value)


def load(path, identity, device='cpu'):
    value = load_checkpoint(path,identity,device)
    learners = ([value['learner']] if 'learner' in value else value.get('learners', []))
    if len(value['saved_gradients']) != len(learners):
        raise ValueError('gradient checkpoint coverage mismatch')
    for learner, gradients in zip(learners,value['saved_gradients']):
        if set(gradients) != dict(learner.model.named_parameters()).keys():
            raise ValueError('gradient checkpoint parameter mismatch')
        for name,p in learner.model.named_parameters():
            p.grad = None if gradients[name] is None else gradients[name].to(p.device).clone()
    return value


def state_payload(identity, learner, metadata, anchor, record):
    return dict(identity=identity, learner=learner, metadata=metadata, anchor=anchor, record=record)


def get_bundle(directory, learner, metadata, anchor, seed, index, config, identity):
    path = directory/f'assessment_{index}'/'candidates.pt'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        value = load(path, identity, str(learner.device))
        if value['parent_signature'] != state_signature(learner):
            raise ValueError('candidate parent changed')
        return value
    ids, replacement = candidate_ids(learner.memory.ids, learner.memory.seen-1,
                                    config['candidate_count'], seed, index)
    by_id = dict(zip(learner.memory.ids, learner.memory.packets))
    ordered = ids+[replacement]
    value = dict(identity=identity, parent_signature=state_signature(learner),
                 parent_steps=learner.cost['optimizer_steps'], anchor=copy.deepcopy(anchor),
                 anchor_id=learner.memory.seen-1, ids=ordered,
                 packets=[copy.deepcopy(by_id[i]) for i in ordered],
                 metadata=[copy.deepcopy(metadata[i]) for i in ordered],
                 reservoir_ids=list(learner.memory.ids))
    value['query_support_overlaps'] = [[i, j] for i in ordered for j in ordered if i != j
        and by_id[j].support is not None and by_id[i].query.fingerprint() == by_id[j].support.fingerprint()]
    save(path, value)
    return value


def get_shadows(path, parent, anchor, bundle, config, identity):
    if path.exists():
        payload = load(path, identity, str(parent.device))
        if (payload['parent_signature'] != state_signature(parent)
                or payload['anchor_sha256'] != packet_fingerprint(anchor)
                or payload['ids'] != bundle['ids']):
            raise ValueError('shadow parent or packet set changed')
        if [state_signature(m) for m in payload['learners']] != payload['signatures']:
            raise ValueError('saved shadow state changed')
        return payload
    signature = state_signature(parent)
    models, work = [], []
    for packet in bundle['packets']:
        model, counters = make_shadow(parent, anchor, packet, config['rehearsal_updates'])
        models.append(model)
        work.append(counters)
    if state_signature(parent) != signature:
        raise AssertionError('rehearsal changed ordinary learner')
    payload = dict(identity=identity, parent_signature=signature,
                   anchor_sha256=packet_fingerprint(anchor), ids=bundle['ids'],
                   learners=models, work=work,
                   signatures=[state_signature(model) for model in models])
    save(path, payload)
    return payload


def prediction_record(probabilities, data, truth=None):
    selected = probabilities[np.arange(len(data)), data.actions].astype(np.float64)
    result = dict(probabilities=selected.tolist(), prediction_sha256=array_hash(probabilities),
                  brier=float(np.square(selected-data.survival).mean()))
    if truth is not None:
        actions = probabilities[:, :, -1].argmax(axis=1)
        result.update(actions=actions.tolist(), survival=float(truth[np.arange(len(data)), actions, -1].mean()))
    return result


def aggregate_panels(rows, key):
    values = [row[key] for row in rows]
    return dict(brier=np.mean([[item['brier'] for item in group] for group in values], axis=0).tolist(),
                survival=np.mean([[item['survival'] for item in group] for group in values], axis=0).tolist(),
                early_brier=[item['brier'] for item in values[0]],
                late_brier=np.mean([[item['brier'] for item in group] for group in values[1:]], axis=0).tolist())


def run_phase(config, identity, seed, model, phase, parent, metadata, anchor, directory):
    folder = directory/phase['name']
    folder.mkdir(exist_ok=True)
    final_path, checkpoint = folder/'result.json', folder/'checkpoint.pt'
    if final_path.exists():
        saved = load(checkpoint, identity, str(parent.device))
        if saved['record'] != read_json(final_path):
            raise ValueError('completed phase record mismatch')
        if saved['record']['before_signature'] != state_signature(parent):
            raise ValueError('phase parent chain changed')
        if saved['record']['final_signature'] != state_signature(saved['learner']):
            raise ValueError('completed learner differs from recorded state')
        return saved['learner'], saved['metadata'], saved['anchor'], saved['record']
    if not (folder/'before.pt').exists():
        record = dict(identity=identity, seed=seed, model=model, phase=phase,
                      before_signature=state_signature(parent), start_diagnostics=parent.diagnostics(),
                      start_memory=memory_state(parent.memory), packets=[], evaluations=[],
                      completed_packets=0, shadow_work=[], frozen_work=[])
        save(folder/'before.pt', state_payload(identity,parent,metadata,anchor,record))
    before = load(folder/'before.pt', identity, str(parent.device))
    if before['record']['before_signature'] != state_signature(parent):
        raise ValueError('phase before checkpoint chain mismatch')
    initial = before['learner']
    if state_signature(initial) != before['record']['before_signature']:
        raise ValueError('saved initial learner changed')
    shadows, frozen, bundle = None, None, None
    if phase['kind'] in ('score', 'near', 'return'):
        index = phase['assessment']
        if phase['kind'] == 'score':
            bundle = get_bundle(directory,initial,before['metadata'],before['anchor'],seed,index,config,identity)
        else:
            bundle = load(directory/f'assessment_{index}'/'candidates.pt',identity,str(parent.device))
            choice = read_json(directory/f'assessment_{index}'/'choice.json')
            score_record = read_json(directory/f'score_{index}'/'result.json')
            expected = select(observable_candidates(bundle['metadata'][:-1]), score_record['score_losses'])
            if (choice['choice'] != expected or choice['identity'] != identity
                    or choice['score_record_sha256'] != digest(score_record)):
                raise ValueError('sealed choice changed')
            frozen = load(directory/f'score_{index}'/'shadows.pt',identity,str(parent.device))
        shadows = get_shadows(folder/'shadows.pt',initial,before['anchor'],bundle,config,identity)
    saved = load(checkpoint,identity,str(parent.device)) if checkpoint.exists() else before
    learner, metadata, anchor, record = (saved[k] for k in ('learner','metadata','anchor','record'))
    if shadows is not None:
        record['shadow_work'] = shadows['work']
    world, law = AcquisitionWorld(), law_from_record(phase['law'])
    size = config['batch_size']
    for index in range(record['completed_packets'], phase['size']//size):
        channel = 'predictive_value_'+phase['name']
        data = batch(world,law,seed,channel,index,size)
        support = learner.history
        probabilities = predict_all(learner,data.observations,support)
        prequential = prediction_record(probabilities,data)
        packet_id = learner.memory.seen
        info = dict(id=packet_id,original_brier=prequential['brier'],
                    original_steps=learner.cost['optimizer_steps'],original_model_sha256=state_hash(learner.model.state_dict()),
                    original_prediction_sha256=prequential['prediction_sha256'],
                    query_sha256=data.fingerprint(),support_sha256=None if support is None else support.fingerprint(),
                    origin_law=phase['law'],origin_phase=phase['name'],origin_index=index)
        row = dict(**info, probabilities=prequential['probabilities'],
                   actions=data.actions.tolist(),outcomes=data.survival.tolist())
        if shadows is not None:
            truth = None
            if phase['kind'] != 'score':
                cases = world.dataset(law,size,trial_seed(seed,'acquisition_'+channel,index))
                if not np.array_equal(cases.observations,data.observations):
                    raise AssertionError('evaluation/query mismatch')
                truth = world.counterfactuals(cases)
            evaluation = dict(packet_id=packet_id,query_sha256=data.fingerprint(),support_sha256=info['support_sha256'])
            if truth is not None:
                evaluation['truth'] = truth.tolist()
            evaluation['reapplied'] = [prediction_record(predict_all(m,data.observations,support),data,truth)
                                       for m in shadows['learners']]
            if frozen is not None:
                evaluation['frozen'] = [prediction_record(predict_all(m,data.observations,support),data,truth)
                                        for m in frozen['learners']]
            record['evaluations'].append(evaluation)
        anchor = Packet(support,data)
        learner.train(data)
        metadata[packet_id] = info
        metadata = {i:metadata[i] for i in learner.memory.ids}
        record['packets'].append(row)
        record['completed_packets'] = index+1
        if (index+1)*size % config['checkpoint_every'] == 0:
            save(checkpoint,state_payload(identity,learner,metadata,anchor,record))
    if shadows is not None:
        for item in (shadows, frozen):
            if item is not None and [state_signature(m) for m in item['learners']] != item['signatures']:
                raise AssertionError('prediction mutated a frozen rehearsal state')
    record.update(final_signature=state_signature(learner),diagnostics=learner.diagnostics(),
                  memory=memory_state(learner.memory),metadata={str(i):row for i,row in metadata.items()})
    if phase['kind'] == 'score':
        record['score_losses'] = np.mean([[item['brier'] for item in r['reapplied']]
                                          for r in record['evaluations']],axis=0).tolist()
        choice = dict(identity=identity,seed=seed,model=model,index=phase['assessment'],
                      score_record_sha256=digest(record),choice=select(observable_candidates(bundle['metadata'][:-1]),record['score_losses']))
        write_json(directory/f"assessment_{phase['assessment']}"/'choice.json',choice)
    elif shadows is not None:
        record['panels'] = {key:aggregate_panels(record['evaluations'],key) for key in ('reapplied','frozen')}
    save(checkpoint,state_payload(identity,learner,metadata,anchor,record))
    write_json(final_path,record)
    print(json.dumps(dict(seed=seed,model=model,phase=phase['name'],completed=True)),flush=True)
    return learner,metadata,anchor,record


def assessment_result(directory, index, config, identity):
    bundle = load(directory/f'assessment_{index}'/'candidates.pt',identity)
    choice = read_json(directory/f'assessment_{index}'/'choice.json')
    score = read_json(directory/f'score_{index}'/'result.json')
    near = read_json(directory/f'near_{index}'/'result.json')
    returned = read_json(directory/f'return_{index}'/'result.json')
    if choice['score_record_sha256'] != digest(score):
        raise ValueError('choice/score seal mismatch')
    if (choice['identity'] != identity
            or choice['choice'] != select(observable_candidates(bundle['metadata'][:-1]),score['score_losses'])):
        raise ValueError('assessment selection differs from past-only score')
    panels = {f'{name}_{kind}':record['panels'][kind]
              for name,record in (('near',near),('return',returned))
              for kind in ('reapplied','frozen')}
    phase_names = [f'{name}_{index}' for name in ('score','near','gap','return')]
    return dict(identity=identity,seed=score['seed'],model=score['model'],index=index,
        cue=score['phase']['cue'],gap=score['phase']['gap'],
        candidates=bundle['metadata'][:-1],replacement=bundle['metadata'][-1],
        reservoir_ids=bundle['reservoir_ids'],anchor_id=bundle['anchor_id'],
        parent_steps=bundle['parent_steps'],query_support_overlaps=bundle['query_support_overlaps'],
        candidate_storage_bytes=sum(p.nbytes() for p in bundle['packets']),
        metadata_storage_bytes=len(json.dumps(bundle['metadata']).encode()),
        choice=choice['choice'],panels=panels,
        phase_hashes={name:hashlib.sha256((directory/name/'result.json').read_bytes()).hexdigest()
                      for name in phase_names},
        work={name:record['shadow_work'] for name,record in (('score',score),('near',near),('return',returned))},
        prediction_work=dict(score_shadow_forwards=(config['candidate_count']+1)*config['score_size']//config['batch_size'],
            validation_shadow_forwards=4*(config['candidate_count']+1)*config['validation_size']//config['batch_size'],
            query_records_per_forward=config['batch_size'],support_records_per_forward=config['batch_size']))


def run_job(config, identity, seed, model, output, device='cpu'):
    actual = affinity(config['cpu_affinity'])
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = Path(output)/f'{model}_{seed}'
    directory.mkdir(exist_ok=True)
    learner = new_learner(model,seed,config,device)
    metadata, anchor = {}, None
    phases = schedule(seed,config)
    for phase in phases:
        learner,metadata,anchor,_ = run_phase(config,identity,seed,model,phase,learner,metadata,anchor,directory)
        if phase['kind'] == 'return':
            record = assessment_result(directory,phase['assessment'],config,identity)
            write_json(directory/f"assessment_{phase['assessment']}"/'result.json',record)
    result = dict(identity=identity,seed=seed,model=model,affinity=actual,
        phase_names=[p['name'] for p in phases],
        phase_hashes={p['name']:hashlib.sha256((directory/p['name']/'result.json').read_bytes()).hexdigest() for p in phases},
        assessment_hashes={str(i):hashlib.sha256((directory/f'assessment_{i}'/'result.json').read_bytes()).hexdigest() for i in (0,1)},
        diagnostics=learner.diagnostics(),final_signature=state_signature(learner),
        prequential_work=dict(forwards=learner.cost['arrivals']//config['batch_size'],
            query_presentations=learner.cost['arrivals'],support_presentations=learner.cost['arrivals']))
    write_json(directory/'result.json',result)
    print(json.dumps(dict(seed=seed,model=model,trajectory_completed=True)),flush=True)
    return dict(seed=seed,model=model,affinity=actual)


def locked_analysis_files(config_path, protocol):
    files = list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('test_predictive*.py'))
    # Portable packaging verification is developed independently and sealed
    # separately; it cannot alter the locked scientific analysis or decisions.
    packaging_only = {'verify_predictive_value_archive.py', 'test_predictive_value_archive.py'}
    files = [p for p in files if p.name not in packaging_only]
    if config_path:
        files.append(Path(config_path).resolve())
    if protocol:
        files.append(Path(protocol).resolve())
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def run_suite(config, output, device='cpu', resume=False, protocol=None, lock_only=False, config_path=None):
    validate_config(config)
    actual = affinity(config['cpu_affinity'])
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    sources = source_manifest()
    runtime = dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,
        numpy=np.__version__,device=device,threads=config['threads'],workers=config['workers'],
        deterministic=True,affinity=actual)
    identity = dict(config_sha256=digest(config),source_sha256=digest(sources),runtime_sha256=digest(runtime))
    if (output/'manifest.json').exists():
        manifest = read_json(output/'manifest.json')
        if not resume or manifest['identity'] != identity or manifest['config'] != config or manifest['source_files'] != sources or manifest['runtime'] != runtime:
            raise ValueError('resume requires exact source/config/runtime')
        with zipfile.ZipFile(output/'training_source.zip') as archive:
            if (set(archive.namelist()) != set(sources) or len(archive.namelist()) != len(sources)
                    or archive.testzip() is not None
                    or any(hashlib.sha256(archive.read(n)).hexdigest() != h for n,h in sources.items())):
                raise ValueError('training-source archive mismatch')
        locked = verify_analysis(output)
        if locked['config'] != config or locked['config_sha256'] != identity['config_sha256']:
            raise ValueError('analysis config mismatch')
        protocol_lock = read_json(output/'protocol_lock.json')
        if (any(protocol_lock[k] != v for k,v in identity.items())
                or hashlib.sha256((output/'protocol_at_lock.md').read_bytes()).hexdigest() != protocol_lock['protocol_sha256']
                or (protocol is not None and Path(protocol).read_bytes() != (output/'protocol_at_lock.md').read_bytes())):
            raise ValueError('protocol changed after locking')
    else:
        if any(output.iterdir()) or protocol is None:
            raise ValueError('new run requires empty output and protocol')
        now = datetime.now(timezone.utc).isoformat()
        manifest = dict(identity=identity,config=config,runtime=runtime,source_files=sources,created_utc=now)
        with zipfile.ZipFile(output/'training_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(ROOT/'src'/name,name)
        files = locked_analysis_files(config_path,protocol)
        with zipfile.ZipFile(output/'analysis_at_lock.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                archive.write(ROOT/name,name)
        write_json(output/'analysis_lock.json',dict(locked_utc=now,files=files,config=config,config_sha256=digest(config)))
        contents = Path(protocol).read_bytes()
        (output/'protocol_at_lock.md').write_bytes(contents)
        write_json(output/'protocol_lock.json',dict(**identity,locked_utc=now,protocol_sha256=hashlib.sha256(contents).hexdigest()))
        write_json(output/'manifest.json',manifest)
    if lock_only:
        result = dict(locked=True,identity=identity,**counts(config))
        print(json.dumps(result,indent=2))
        return result
    if (output/'completion.json').exists():
        result = read_json(output/'completion.json')
        if result['identity'] != identity:
            raise ValueError('completion identity mismatch')
        return result
    jobs = [(seed,model) for model in config['models'] for seed in config['seeds']]
    finished = []
    if config['workers'] == 1:
        for seed,model in jobs:
            finished.append(run_job(config,identity,seed,model,output,device))
    else:
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            pending = [pool.submit(run_job,config,identity,seed,model,output,device) for seed,model in jobs]
            for future in as_completed(pending):
                finished.append(future.result())
    if source_manifest() != sources:
        raise ValueError('source changed during fitting')
    verify_analysis(output)
    completion = dict(identity=identity,**counts(config),jobs=sorted(finished,key=lambda j:(j['model'],j['seed'])),
                      completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'completion.json',completion)
    print(json.dumps(dict(completed=True,**counts(config)),indent=2))
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--lock-only',action='store_true')
    args = parser.parse_args()
    run_suite(read_json(args.config),args.output,args.device,args.resume,args.protocol,args.lock_only,args.config)
