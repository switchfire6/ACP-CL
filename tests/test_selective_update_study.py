"""Study-level budget, fork, locking and interrupted-execution checks."""

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_evidence_consolidation import semantic
from acp_cl.selective_updates import study
from acp_cl.replay_renewal import study as original_study


@pytest.fixture
def settings():
    torch.set_num_threads(1)
    return json.loads(Path("configs/selective_updates_smoke.json").read_text()) | {
        "models": ["conditional"], "prefix_size": 32, "episode_size": 32,
        "probe_every": 16, "width": 4, "decoder_width": 4, "context_width": 3,
    }


def test_lock_precedes_fitting_and_rejects_configuration_change(settings, tmp_path):
    output = tmp_path/"locked"
    result = study.run_suite(settings, output, lock_only=True)
    assert result["locked"] and result["episodes"] == 6
    assert (output/"analysis_at_lock.zip").exists()
    assert (output/"training_source.zip").exists()
    assert not list(output.glob("*/*.pt"))
    with pytest.raises(ValueError, match="identical config/source/runtime"):
        study.run_suite(dict(settings, lr=.003), output, resume=True, lock_only=True)
    with pytest.raises(ValueError, match="identical config/source/runtime"):
        study.run_suite(settings, output, lock_only=True)


def test_full_local_cohort_resumes_after_an_atomic_checkpoint(settings, tmp_path, monkeypatch):
    complete, interrupted = tmp_path/"complete", tmp_path/"interrupted"
    finished = study.run_suite(settings, complete)
    assert finished["episodes"] == 6 and finished["prefix_fits"] == 1
    original = original_study.atomic_checkpoint
    stopped = False

    def stop_after_publishing(path, payload):
        nonlocal stopped
        original(path, payload)
        if Path(path).name == "checkpoint.pt" and Path(path).parent.name == "stage_1_combined" and not stopped:
            stopped = True
            raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(original_study, "atomic_checkpoint", stop_after_publishing)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        study.run_suite(settings, interrupted)
    monkeypatch.setattr(original_study, "atomic_checkpoint", original)
    resumed = study.run_suite(settings, interrupted, resume=True)
    assert resumed["episodes"] == 6
    for path in complete.glob("*/stage_1_*/result.json"):
        a = json.loads(path.read_text())
        b = json.loads((interrupted/path.relative_to(complete)).read_text())
        assert semantic(a) == semantic(b)
        before, after = a["start_diagnostics"], a["diagnostics"]
        assert after["cost"]["arrivals"]-before["cost"]["arrivals"] == settings["episode_size"]
        assert after["selective_updates"]["stats"]["steps"] == settings["episode_size"]//settings["batch_size"]
    assert study.run_suite(settings, interrupted, resume=True) == resumed


def test_protocol_bytes_cannot_change_on_resume(settings, tmp_path):
    # Protocols in real runs live in the repository and are included in the
    # analysis ZIP. This test supplies an existing repository protocol.
    output = tmp_path/"locked"
    protocol = Path("docs/selective_updates_protocol.md")
    study.run_suite(settings, output, protocol=protocol, lock_only=True)
    (output/"protocol_at_lock.md").write_bytes(b"tampered copy")
    with pytest.raises(ValueError, match="protocol changed"):
        study.run_suite(settings, output, protocol=protocol, resume=True, lock_only=True)


def test_analysis_archive_tampering_is_rejected(settings, tmp_path):
    output = tmp_path/"locked"
    study.run_suite(settings, output, lock_only=True)
    lock_path = output/"analysis_lock.json"
    lock = json.loads(lock_path.read_text())
    name = next(iter(lock["files"]))
    lock["files"][name] = "0"*64
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="analysis changed after locking"):
        study.verify_analysis(output)
