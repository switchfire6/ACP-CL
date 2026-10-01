"""Verify a portable sealed archive without loading model checkpoints."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

from acp_cl.persistence.study import digest
from summarize_replay_renewal import expected_keys, record_path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_zip(path, expected):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or set(z.namelist()) != set(expected):
            raise ValueError(f"archive contents differ: {path}")
        for name, sha in expected.items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError(f"archived source mismatch: {name}")


def verify(directory):
    directory = Path(directory).resolve()
    artifacts = read(directory/"artifact_manifest.json")
    files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}
    if set(artifacts) != files:
        raise ValueError("artifact file set changed")
    for name, record in artifacts.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("artifact path escapes archive")
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"artifact mismatch: {name}")
    final_sources = read(directory/"analysis_source_manifest.json")
    prospective = read(directory/"analysis_at_lock.json")["files"]
    verify_zip(directory/"analysis_source.zip", final_sources)
    verify_zip(directory/"analysis_at_lock.zip", prospective)
    if any(final_sources.get(name) != value for name, value in prospective.items()):
        raise ValueError("prospective analysis/config/protocol changed after lock")
    cohorts, identities = {}, []
    for experiment in ("decomposition", "policy"):
        folder = directory/experiment
        manifest, summary = read(folder/"manifest.json"), read(folder/"summary.json")
        config, identity = manifest["config"], manifest["identity"]
        for name, value in (("config_sha256", config), ("source_sha256", manifest["source_files"]),
                            ("runtime_sha256", manifest["runtime"])):
            if digest(value) != identity[name]:
                raise ValueError("manifest identity mismatch")
        verify_zip(folder/"training_source.zip", manifest["source_files"])
        protocol = read(folder/"protocol_lock.json")
        if hashlib.sha256((folder/"protocol_at_lock.md").read_bytes()).hexdigest() != protocol["protocol_sha256"]:
            raise ValueError("protocol mismatch")
        if summary["identity"] != identity or any(protocol[k] != v for k, v in identity.items()):
            raise ValueError("summary/lock identity mismatch")
        hashes = {r["path"]: r["sha256"] for r in summary["result_hashes"]}
        seen = set()
        with gzip.open(folder/"raw_results.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                key = record["seed"], record["model"], record["stage"], record["arm"], record["branch"]
                if key in seen or record["identity"] != identity:
                    raise ValueError("duplicate or mismatched raw record")
                seen.add(key)
                name = (record_path(Path("."), config, key)/"result.json").as_posix()
                original_bytes = (json.dumps(record, indent=2, allow_nan=False)+"\n").encode("utf-8")
                if hashlib.sha256(original_bytes).hexdigest() != hashes[name]:
                    raise ValueError("raw record no longer matches original result bytes")
        if seen != expected_keys(config) or len(hashes) != len(seen):
            raise ValueError("incomplete or extra raw records")
        audit, integrity = read(folder/"audit.json"), read(folder/"file_integrity.json")
        if not audit["passed"] or not integrity["passed"] or audit["identity"] != identity:
            raise ValueError("failed or mismatched verification")
        identities.append(identity["source_sha256"])
        cohorts[experiment] = dict(episodes=len(seen), prefix_checkpoints=audit["prefix_checkpoints"],
            before_checkpoints=audit["before_checkpoints"], final_checkpoints=audit["final_checkpoints"],
            file_checks=len(integrity["files"]))
    if len(set(identities)) != 1:
        raise ValueError("cohorts used different training source")
    result = dict(passed=True, artifacts=len(artifacts), bytes=sum(r["bytes"] for r in artifacts.values()),
        prospective_analysis_unchanged=True, cohorts=cohorts)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    verify(args.input)
