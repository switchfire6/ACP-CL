"""Verify sealed portable records, locks and original JSON hashes without pickle."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

from acp_cl.evidence_consolidation.study import counts
from acp_cl.persistence.study import digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(expected):
            raise ValueError(f"source ZIP file-set/CRC mismatch: {path}")
        for name, sha in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError(f"source ZIP hash mismatch: {name}")


def original_sha(record):
    return hashlib.sha256((json.dumps(record, indent=2, allow_nan=False)+"\n").encode("utf-8")).hexdigest()


def verify(directory):
    directory = Path(directory).resolve()
    artifacts = read(directory/"artifact_manifest.json")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p.name != "artifact_manifest.json"}
    if set(artifacts) != actual:
        raise ValueError("portable artifact file set changed")
    for name, item in artifacts.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("artifact path escapes directory")
        content = path.read_bytes()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"artifact changed: {name}")
    sources = read(directory/"analysis_source_manifest.json")
    locked = read(directory/"analysis_at_lock.json")["files"]
    verify_zip(directory/"analysis_source.zip", sources)
    verify_zip(directory/"analysis_at_lock.zip", locked)
    if any(sources.get(name) != sha for name, sha in locked.items()):
        raise ValueError("prospective analysis changed after locking")
    cohorts, training_sources = {}, []
    protocol_sha = locked["docs/evidence_consolidation_protocol.md"]
    for name in ("development", "pilot"):
        folder = directory/name
        manifest, summary = read(folder/"manifest.json"), read(folder/"summary.json")
        config, identity = manifest["config"], manifest["identity"]
        if config != read(folder/"config.json") or summary["config"] != config:
            raise ValueError("portable cohort configuration mismatch")
        config_path = f"configs/evidence_consolidation_{name}.json"
        with zipfile.ZipFile(directory/"analysis_at_lock.zip") as archive:
            if json.loads(archive.read(config_path)) != config:
                raise ValueError("cohort configuration differs from prospective lock")
        for key, value in (("config_sha256", config), ("source_sha256", manifest["source_files"]),
                           ("runtime_sha256", manifest["runtime"])):
            if digest(value) != identity[key]:
                raise ValueError("manifest identity mismatch")
        verify_zip(folder/"training_source.zip", manifest["source_files"])
        lock = read(folder/"protocol_lock.json")
        if any(lock[k] != v for k, v in identity.items()) or hashlib.sha256(
                (folder/"protocol_at_lock.md").read_bytes()).hexdigest() != lock["protocol_sha256"]:
            raise ValueError("protocol lock mismatch")
        if lock["protocol_sha256"] != protocol_sha:
            raise ValueError("cohort protocol differs from prospective analysis lock")
        if summary["identity"] != identity:
            raise ValueError("summary identity mismatch")
        pairs = [(s, m) for s in config["seeds"] for m in config["models"]]
        expected = {f"{m}_{s}/block_{i:02}/result.json" for s, m in pairs for i in range(config["blocks"])}
        expected |= {f"{m}_{s}/fresh_{i}/result.json" for s, m in pairs for i in (1, 2, 3)}
        expected |= {f"{m}_{s}/prefix.json" for s, m in pairs}
        hashes = {r["path"]: r["sha256"] for r in summary["result_hashes"]}
        if set(hashes) != expected or len(hashes) != len(summary["result_hashes"]):
            raise ValueError("record hash list incomplete or duplicated")
        found = {}
        with gzip.open(folder/"raw_results.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                tag = f"block_{record['block']['index']:02}" if "block" in record else f"fresh_{record['stage']}"
                path = f"{record['model']}_{record['seed']}/{tag}/result.json"
                if path in found or record["identity"] != identity:
                    raise ValueError("duplicate/mismatched raw record")
                found[path] = original_sha(record)
        for tag, record in read(folder/"prefix_records.json").items():
            path = f"{tag}/prefix.json"
            if tag != f"{record['model']}_{record['seed']}" or record["identity"] != identity:
                raise ValueError("prefix identity mismatch")
            found[path] = original_sha(record)
        if found != hashes:
            raise ValueError("portable records differ from original JSON hashes")
        completion = read(folder/"completion.json")
        if completion["identity"] != identity or any(completion[k] != v for k, v in counts(config).items()):
            raise ValueError("completion metadata mismatch")
        audit, integrity = read(folder/"audit.json"), read(folder/"file_integrity.json")
        if not audit["passed"] or not integrity["passed"] or audit["identity"] != identity:
            raise ValueError("cohort did not pass its audit")
        training_sources.append(identity["source_sha256"])
        cohorts[name] = dict(**counts(config), decision_windows=audit["decision_windows"],
                             inspected_files=len(integrity["files"]))
    if len(set(training_sources)) != 1:
        raise ValueError("cohorts used different training source")
    result = dict(passed=True, artifacts=len(artifacts), prospective_analysis_unchanged=True, cohorts=cohorts)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    verify(parser.parse_args().input)
