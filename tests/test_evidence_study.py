"""Whole-run source locking, causal audit and interruption inside evidence windows."""

import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_evidence_consolidation import (advance_lineage, audit, semantic, signature)
from acp_cl.evidence_consolidation import study
from acp_cl.evidence_consolidation.mechanism import DELAYED


@pytest.mark.parametrize("model,window", (("conditional", 2), ("recurrent", 3)))
def test_full_smoke_audit_and_exact_midwindow_recovery(model, window, tmp_path, monkeypatch):
    config = json.loads(Path("configs/evidence_consolidation_smoke.json").read_text())
    config.update(models=[model], evidence_window=window)
    protocol = tmp_path/"protocol.md"
    protocol.write_text("Engineering recovery protocol; no scientific qualification.\n")
    complete, interrupted = tmp_path/"complete", tmp_path/"interrupted"
    study.run_suite(config, complete, protocol=protocol)
    original = study.atomic_checkpoint
    interrupted_state = {}

    def crash(path, payload):
        original(path, payload)
        record = payload["record"]
        if "block" in record and record["block"]["index"] == 0 and record["batches_done"] == 1:
            bundle = payload["bundle"]
            assert bundle.proposal is not None
            assert 0 < len(bundle.gains["sustained"]) < window
            interrupted_state["signature"] = signature(bundle)
            interrupted_state["path"] = path
            raise RuntimeError("simulated midwindow interruption")

    monkeypatch.setattr(study, "atomic_checkpoint", crash)
    with pytest.raises(RuntimeError, match="simulated midwindow"):
        study.run_suite(config, interrupted, protocol=protocol)
    saved = torch.load(interrupted_state["path"], weights_only=False)
    assert signature(saved["bundle"]) == interrupted_state["signature"]
    monkeypatch.setattr(study, "atomic_checkpoint", original)
    study.run_suite(config, interrupted, resume=True, protocol=protocol)

    left = sorted(complete.glob("*/*/result.json"))
    right = sorted(interrupted.glob("*/*/result.json"))
    assert len(left) == len(right) == 6  # Three complete stream blocks and three fresh references.
    for path in left:
        counterpart = interrupted/path.relative_to(complete)
        assert semantic(json.loads(path.read_text())) == semantic(json.loads(counterpart.read_text()))
    for path in complete.glob("*/block_*/*.pt"):
        a = torch.load(path, weights_only=False)
        b = torch.load(interrupted/path.relative_to(complete), weights_only=False)
        assert signature(a["bundle"]) == signature(b["bundle"])
        assert semantic(a["record"]) == semantic(b["record"])
    result = audit(interrupted, tmp_path/"audit.json")
    assert result["passed"]
    assert result["prefix_checkpoints"] == 1
    assert result["block_before_checkpoints"] == result["block_final_checkpoints"] == 3
    assert result["fresh_before_checkpoints"] == result["fresh_final_checkpoints"] == 3
    assert result["decision_windows"] == 10//window
    with pytest.raises(ValueError, match="identical"):
        study.run_suite(dict(config, evidence_margin=.003), interrupted, resume=True, protocol=protocol)
    protocol.write_text("Changed protocol.\n")
    with pytest.raises(ValueError, match="protocol changed"):
        study.run_suite(config, interrupted, resume=True, protocol=protocol)


@pytest.mark.parametrize("fault", ("acceptance", "lineage", "window"))
def test_auditor_rejects_inconsistent_gate_logs(fault):
    original, proposal = "a"*64, "a"*64
    chain = dict(initial=original, end=0, committed={p: original for p in DELAYED},
                 adoptions={p: 0 for p in DELAYED})
    decision = dict(start_batch=0, end_batch=2, proposal_hash=proposal,
        committed_before=copy.deepcopy(chain["committed"]),
        gains={p: [0., 0.] for p in DELAYED},
        accepted=dict(periodic=True, single=False, sustained=False))
    config = dict(evidence_window=2, evidence_margin=.002, evidence_positive=2)
    advance_lineage(copy.deepcopy(chain), [decision], config)
    if fault == "acceptance":
        decision["accepted"]["periodic"] = False
    elif fault == "lineage":
        decision["committed_before"]["sustained"] = "b"*64
    else:
        decision["start_batch"] = 1
    with pytest.raises(AssertionError):
        advance_lineage(chain, [decision], config)
