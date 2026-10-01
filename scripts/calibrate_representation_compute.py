"""Allocate useful outcome-only updates using engineering timing and matrix work.

This script never evaluates predictions or reads scientific result artifacts.
The single fixed engineering measurement is recorded even if it fails, and an
existing attempt cannot be silently replaced by a more favorable timing run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

from acp_cl.acquisition.world import AcquisitionWorld, Law
from acp_cl.conditional.learner import Packet
from acp_cl.persistence.study import digest, trial_seed
from acp_cl.persistence.world import IMAGE_SHAPE
from acp_cl.replay_renewal.memory import memory_state
from acp_cl.replay_renewal.study import affinity, write_json
from acp_cl.representation_learning.design import validate_config
from acp_cl.representation_learning.learner import RepresentationLearner


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING_SEED = 18091
TIMING_ARMS = ('outcome', 'final_frame', 'sequence')
ROUNDS = 5
WARMUP_BATCHES = 16
MEASURED_BATCHES = 8
WORK_FIELDS = (
    'arrivals', 'optimizer_steps', 'query_presentations', 'support_presentations',
    'replay_presentations', 'duplicate_presentations', 'encoder_forward_calls',
    'encoder_presentations', 'auxiliary_forward_calls', 'auxiliary_reconstructions',
    'reconstruction_pixels', 'auxiliary_backward_steps', 'predictive_parameter_updates',
    'auxiliary_parameter_updates',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    """Hash imported project dependencies, excluding unrelated active work."""
    paths = {Path(__file__).resolve()}
    for name, module in list(sys.modules.items()):
        value = getattr(module, '__file__', None)
        if name.startswith('acp_cl') and value:
            path = Path(value).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT / 'src'):
                paths.add(path)
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def round_order(index):
    return TIMING_ARMS if index % 2 == 0 else tuple(reversed(TIMING_ARMS))


def dense_mac_proxy(config):
    """Dense matrix multiply-accumulates; not exact device FLOPs.

    Each update trains current and replay packets. Each packet jointly encodes
    B support and B query sequences, each with four frames. The context GRU
    processes B support events; only B query latents enter either decoder.
    A common factor of three approximates forward plus backward matrix work.
    Biases, nonlinearities, interaction products, Adam and clipping are omitted
    from this proxy and remain present in the timing measurement.
    """
    batch, width = config['batch_size'], config['width']
    context, decoder = config['context_width'], config['decoder_width']
    frames, height, columns = IMAGE_SHAPE
    event_width = width + 8 + (5 * width if config['interaction_features'] else 0)
    packet = dict(
        frame_linear=frames * (2 * batch) * (height * columns) * width,
        temporal_gru=frames * (2 * batch) * 3 * (width * width + width * width),
        context_gru=batch * 3 * (event_width * context + context * context),
        outcome_decoder=batch * ((width + context) * decoder + decoder * 15),
    )
    auxiliary = batch * (width * width + width * math.prod(IMAGE_SHAPE))
    shared_update = 2 * sum(packet.values())
    auxiliary_update = 2 * auxiliary
    return dict(
        units='dense multiply-accumulates; one multiply plus accumulate counted as one',
        packet_forward_components=packet, auxiliary_query_forward=auxiliary,
        packets_per_update=2, forward_backward_multiplier=3,
        outcome_training_per_update=3 * shared_update,
        auxiliary_training_per_update=3 * (shared_update + auxiliary_update),
        auxiliary_base_ratio=(shared_update + auxiliary_update) / shared_update,
        excluded='biases, nonlinearities, interactions, loss elementwise work, clipping, Adam',
    )


def allocate_updates(base_updates, median_seconds, mac_ratio):
    if type(base_updates) is not int or base_updates < 1:
        raise ValueError('base updates must be a positive integer')
    if set(median_seconds) != set(TIMING_ARMS):
        raise ValueError('all three timing arms are required')
    if any(not math.isfinite(v) or v <= 0 for v in median_seconds.values()):
        raise ValueError('timings must be finite and positive')
    if not math.isfinite(mac_ratio) or mac_ratio <= 0:
        raise ValueError('matrix-work ratio must be finite and positive')
    ratios = {a: median_seconds[a] / median_seconds['outcome'] for a in TIMING_ARMS[1:]}
    multiplier = max(1., *ratios.values(), mac_ratio)
    return dict(
        compute_updates_per_batch=math.ceil(base_updates * multiplier),
        base_updates_per_batch=base_updates, timing_ratios=ratios,
        maximum_auxiliary_timing_ratio=max(ratios.values()),
        matrix_work_ratio=mac_ratio, unrounded_multiplier=multiplier,
        rule='ceil(base_updates * max(1, both auxiliary/base ratios of medians, MAC ratio))',
    )


def work_snapshot(learner):
    """A whitelist prevents any training loss or evaluation score being exported."""
    return {name: int(learner.cost[name]) for name in WORK_FIELDS}


def resource_snapshot(learner):
    predictive = [p for name, p in learner.model.named_parameters()
                  if not name.startswith('auxiliary.')]
    auxiliary = [p for name, p in learner.model.named_parameters()
                 if name.startswith('auxiliary.')]
    return dict(
        predictive_parameters=sum(p.numel() for p in predictive),
        auxiliary_parameters=sum(p.numel() for p in auxiliary),
        predictive_parameter_bytes=sum(p.numel() * p.element_size() for p in predictive),
        auxiliary_parameter_bytes=sum(p.numel() * p.element_size() for p in auxiliary),
        gradient_bytes=sum(p.grad.numel() * p.grad.element_size()
                           for p in learner.model.parameters() if p.grad is not None),
        optimizer_tensor_bytes=sum(v.numel() * v.element_size()
                                  for state in learner.optimizer.state.values()
                                  for v in state.values() if isinstance(v, torch.Tensor)),
        replay_payload_bytes=learner.memory.nbytes(),
        peak_replay_payload_bytes=learner.peak_memory_bytes,
        history_payload_bytes=Packet(None, learner.history).nbytes(),
        replay_packets=len(learner.memory.packets), replay_capacity=learner.memory.capacity,
        accounting='tensor and packet payload bytes, not Python overhead or peak process RSS',
    )


def replay_signature(learner):
    return digest(dict(memory=memory_state(learner.memory),
                       history=learner.history.fingerprint()))


def measure_round(config, packets, index):
    if len(packets) != WARMUP_BATCHES + MEASURED_BATCHES:
        raise ValueError('the engineering stream has an incorrect fixed length')
    result = dict(index=index, arm_order=list(round_order(index)), arms={})
    for arm in result['arm_order']:
        learner = RepresentationLearner(arm, ENGINEERING_SEED, config)
        for packet in packets[:WARMUP_BATCHES]:
            learner.train(packet)
        if len(learner.memory.packets) != config['memory_packets']:
            raise AssertionError('warmup must fill the replay reservoir')
        warm_replay = replay_signature(learner)
        before = work_snapshot(learner)
        durations = []
        for packet in packets[WARMUP_BATCHES:]:
            started = time.perf_counter()
            learner.train(packet)
            durations.append(time.perf_counter() - started)
        after = work_snapshot(learner)
        result['arms'][arm] = dict(
            measured_batch_seconds=durations, training_seconds=sum(durations),
            initial_predictor_hash=learner.initial_predictor_hash,
            initial_auxiliary_hash=learner.initial_auxiliary_hash,
            warmup_replay_sha256=warm_replay, final_replay_sha256=replay_signature(learner),
            measured_work={k: after[k] - before[k] for k in WORK_FIELDS},
            total_work=after, final_resources=resource_snapshot(learner),
        )
        del learner
    # All methods see the same records, original supports and replay draws.
    for key in ('initial_predictor_hash', 'warmup_replay_sha256', 'final_replay_sha256'):
        if len({row[key] for row in result['arms'].values()}) != 1:
            raise AssertionError(f'paired timing arms differ in {key}')
    return result


def run(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    attempt_path = output.with_name(output.stem + '.attempt.json')
    if output.exists() or attempt_path.exists():
        raise FileExistsError('existing calibration or attempt is preserved; do not retime it')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    if config['smoke'] or config['kind'] != 'qualification':
        raise ValueError('allocation requires the full-size qualification configuration')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    actual_affinity = affinity(config['cpu_affinity'])
    if actual_affinity != 1365:
        raise AssertionError('engineering CPU affinity differs from the fixed mask')
    sources = source_files()
    config_hash = sha(config_path)
    protocol = ROOT / 'docs/representation_learning_protocol.md'
    protocol_hash = sha(protocol)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    attempt = dict(status='started', started_at_utc=started_at,
                   config_sha256=config_hash, source_files=sources,
                   protocol_sha256=protocol_hash, output=str(output), seed=ENGINEERING_SEED)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also guards accidental concurrent runs.
    with attempt_path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(attempt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    rounds = []
    try:
        world = AcquisitionWorld()
        laws = [Law(i % 2, (0, 1, 2)) for i in range(WARMUP_BATCHES + MEASURED_BATCHES)]
        seeds = [trial_seed(ENGINEERING_SEED, 'representation_compute', i)
                 for i in range(len(laws))]
        # All input generation and hashing happen outside the timed calls.
        packets = [world.experience(law, config['batch_size'], seed)
                   for law, seed in zip(laws, seeds, strict=True)]
        fingerprints = [packet.fingerprint() for packet in packets]
        for index in range(ROUNDS):
            row = measure_round(config, packets, index)
            if [p.fingerprint() for p in packets] != fingerprints:
                raise AssertionError('engineering training mutated paired inputs')
            rounds.append(row)
            print(json.dumps(dict(round=index + 1, measured_seconds={
                arm: row['arms'][arm]['training_seconds'] for arm in TIMING_ARMS})), flush=True)
            write_json(attempt_path, dict(attempt, completed_rounds=rounds))
        medians = {arm: statistics.median(row['arms'][arm]['training_seconds'] for row in rounds)
                   for arm in TIMING_ARMS}
        proxy = dense_mac_proxy(config)
        allocation = allocate_updates(config['updates_per_batch'], medians,
                                      proxy['auxiliary_base_ratio'])
        if (source_files() != sources or sha(config_path) != config_hash
                or sha(protocol) != protocol_hash):
            raise AssertionError('calibration source, configuration or protocol changed during timing')
        # Repeated rounds replay exactly the same engineering workload.
        for arm in TIMING_ARMS:
            stable = ('initial_predictor_hash', 'initial_auxiliary_hash',
                      'warmup_replay_sha256', 'final_replay_sha256', 'measured_work',
                      'total_work', 'final_resources')
            if any(any(row['arms'][arm][key] != rounds[0]['arms'][arm][key] for key in stable)
                   for row in rounds[1:]):
                raise AssertionError('repeated timing rounds changed the paired workload')
        record = dict(
            schema_version=1, status='complete', purpose='engineering compute allocation only',
            started_at_utc=started_at, completed_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started, seed=ENGINEERING_SEED,
            source_files=sources, source_files_sha256=digest(sources),
            config_path=str(config_path), config_sha256=config_hash, config=config,
            protocol_sha256=protocol_hash, warmup_batches=WARMUP_BATCHES,
            measured_batches=MEASURED_BATCHES, timing_rounds=ROUNDS,
            initial_state='fresh same predictor and auxiliary initializations for each round',
            input_schedule=[dict(law=asdict(law), seed=seed, query_sha256=fingerprint)
                            for law, seed, fingerprint in zip(laws, seeds, fingerprints, strict=True)],
            paired_input_sha256=digest(fingerprints), rounds=rounds,
            median_training_seconds=medians, mac_proxy=proxy, **allocation,
            runtime=dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                         torch=torch.__version__, numpy=np.__version__, device='cpu',
                         cpu_affinity=actual_affinity, threads=torch.get_num_threads(),
                         interop_threads=torch.get_num_interop_threads(),
                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
            limitations='Engineering timing and a dense-matrix proxy allocate updates; '
                        'actual scientific-run work and time must still be reported. '
                        'No evaluation, prediction quality or training-loss values are recorded.',
        )
        record['allocation_sha256'] = digest(record)
        write_json(output, record)
        write_json(attempt_path, dict(attempt, status='complete', completed_rounds=rounds,
                                     allocation_file_sha256=sha(output)))
        print(json.dumps(dict(output=str(output), **allocation)), flush=True)
        return record
    except BaseException as error:
        write_json(attempt_path, dict(attempt, status='failed', completed_rounds=rounds,
                                     error_type=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
