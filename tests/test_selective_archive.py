"""Portable file-set/record tampering and independent numerical recomputation."""

import copy
import gzip
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import verify_selective_updates_archive as verifier


def seal(directory):
    entries = {path.relative_to(directory).as_posix(): dict(bytes=len(path.read_bytes()), sha256=verifier.sha(path.read_bytes()))
               for path in directory.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"}
    (directory/"artifact_manifest.json").write_text(json.dumps(entries))


@pytest.mark.parametrize("fault", ("bytes", "extra", "missing"))
def test_artifact_manifest_rejects_tampered_or_changed_file_set(tmp_path, fault):
    (tmp_path/"record.json").write_text("{}")
    seal(tmp_path)
    assert verifier.verify_artifacts(tmp_path) == 1
    if fault == "bytes":
        (tmp_path/"record.json").write_text("[]")
    elif fault == "extra":
        (tmp_path/"unlisted.json").write_text("{}")
    else:
        (tmp_path/"record.json").unlink()
    with pytest.raises(ValueError, match="changed"):
        verifier.verify_artifacts(tmp_path)


def raw_fixture(folder):
    config = dict(seeds=[13991], models=["conditional"], prefix_blocks=2,
                  episode_size=16, batch_size=8, probe_every=8)
    identity = dict(example="synthetic")
    records = []
    cue = verifier.ORDERS[13991 % 6][0]
    for arm in verifier.ARMS:
        records.append(dict(identity=identity, seed=13991, model="conditional", arm=arm, stage=1, branch="novel",
            law=dict(mode=0, active=[cue], revised=False, noise=0.), cue=cue, channel="stage_1",
            batches_done=2, batch_sha256=["a"*64, "b"*64], curve=[dict(arrivals=n) for n in (0, 8, 16)]))
    prefixes = {"conditional_13991": dict(identity=identity, seed=13991, model="conditional", policy="uniform", blocks=[{}, {}])}

    def write():
        with gzip.open(folder/"raw_results.jsonl.gz", "wt", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record)+"\n")
        (folder/"prefix_records.json").write_text(json.dumps(prefixes))

    write()
    hashes = [dict(path=f"conditional_13991/stage_1_{record['arm']}/result.json", sha256=verifier.sha(verifier.original_json(record)))
              for record in records]
    hashes.append(dict(path="conditional_13991/prefix.json", sha256=verifier.sha(verifier.original_json(prefixes["conditional_13991"]))))
    return config, identity, dict(result_hashes=hashes), records, prefixes, write


@pytest.mark.parametrize("fault", ("mutated_record", "missing_record", "duplicate_record", "changed_prefix"))
def test_raw_record_roundtrip_rejects_tampering_even_without_outer_manifest(tmp_path, fault):
    config, identity, summary, records, prefixes, write = raw_fixture(tmp_path)
    loaded, _, _ = verifier.read_records(tmp_path, config, identity, summary)
    assert len(loaded) == 6
    if fault == "mutated_record":
        records[0]["batch_sha256"][0] = "c"*64
    elif fault == "missing_record":
        records.pop()
    elif fault == "duplicate_record":
        records.append(copy.deepcopy(records[0]))
    else:
        prefixes["conditional_13991"]["blocks"][0]["unexpected"] = 1
    write()
    with pytest.raises(ValueError):
        verifier.read_records(tmp_path, config, identity, summary)


def test_zip_hashes_require_exact_file_set_and_content(tmp_path):
    path = tmp_path/"source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"original")
    expected = {"source.py": verifier.sha(b"original")}
    verifier.verify_zip(path, expected)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verifier.verify_zip(path, expected)


def test_independent_metrics_respect_uneven_exposure_and_clean_support_semantics():
    record = dict(curve=[dict(arrivals=x, metrics=dict(focus_brier=y, survival=s))
        for x, y, s in ((0, .4, .5), (4, .3, .6), (10, .2, .7))],
        valid_after={"0": dict(valid_brier=.1), "1": dict(valid_brier=.3)},
        after=dict(replicates=[dict(curve=[dict(metrics=dict(brier=.2, focus_brier=.15, marginal_brier=.4))],
                                   flipped=dict(focus_brier=.18)) for _ in range(2)]))
    result = verifier.metrics(record)
    assert result["brier_auc"] == pytest.approx(.29)
    assert result["survival_auc"] == pytest.approx(.61)
    assert result["valid_after"] == pytest.approx(.2)
    assert result["valid_after_mode_0"] == .1 and result["valid_after_mode_1"] == .3
    assert result["cue_effect"] == pytest.approx(.03) and result["marginal_gain"] == pytest.approx(.2)


def test_independent_bootstrap_keeps_six_paired_seed_values():
    values = np.array([-.006, -.003, -.002, -.001, .001, .002])
    seeds = list(range(13001, 13007))
    result = verifier.bootstrap(values, seeds)
    indices = np.random.default_rng(27192026).integers(0, 6, (20000, 6))
    bounds = np.quantile(values[indices].mean(axis=1), [.025, .975])
    assert result["lower"] == bounds[0] and result["upper"] == bounds[1]
    assert result["n"] == 6 and result["positive"] == 2 and result["seeds"] == seeds
    assert result["differences"] == values.tolist()


def test_nonfinite_or_changed_summary_statistics_are_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        verifier.decode('{"value": NaN}')
    with pytest.raises(ValueError, match="numeric mismatch"):
        verifier.compare(dict(mean=.01), dict(mean=.02))
    with pytest.raises(ValueError, match="value mismatch"):
        verifier.compare(True, 1)
