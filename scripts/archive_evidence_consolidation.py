"""Seal portable evidence-consolidation results and preserve prospective analysis."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from acp_cl.replay_renewal.study import write_json


def analysis_files(root):
    return [*sorted((root/"scripts").glob("*.py")),
            *sorted((root/"tests").glob("test_evidence*.py")),
            *sorted((root/"configs").glob("evidence_consolidation*.json")),
            root/"docs/evidence_consolidation_protocol.md"]


def snapshot(output, lock_only=False):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = analysis_files(root)
    if lock_only:
        files = [p for p in files if not p.name.startswith(("plot_", "archive_", "verify_"))]
        if (output/"analysis_at_lock.json").exists():
            raise ValueError("prospective analysis snapshot already exists")
        hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        with zipfile.ZipFile(output/"analysis_at_lock.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, p.relative_to(root).as_posix())
        write_json(output/"analysis_at_lock.json", dict(locked_utc=datetime.now(timezone.utc).isoformat(), files=hashes))
        print(f"Locked {len(files)} prospective analysis/config/protocol files.")
        return
    for cohort in ("development", "pilot"):
        for name in ("audit.json", "file_integrity.json"):
            if not json.loads((output/cohort/name).read_text(encoding="utf-8"))["passed"]:
                raise ValueError("all cohorts must pass audit and integrity checks")
    # Include analysis helpers transitively, including older imported auditors.
    files += sorted((root/"scripts").glob("*.py"))
    files += [root/name for name in ("docs/ADAM_HANDOFF.md", "docs/adam_related_work.md", "docs/evidence_noise_limits.md",
        "docs/research_charter.md", "docs/reproduction.md", "reports/evidence_consolidation_results.md",
        "reports/evidence_consolidation/study_ledger.md", "README.md", "pyproject.toml", "requirements-lock.txt")]
    files = sorted(set(files))
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    frozen = json.loads((output/"analysis_at_lock.json").read_text(encoding="utf-8"))["files"]
    if any(hashes.get(name) != sha for name, sha in frozen.items()):
        raise ValueError("prospective analysis/config/protocol changed")
    with zipfile.ZipFile(output/"analysis_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(root).as_posix())
    write_json(output/"analysis_source_manifest.json", hashes)
    artifacts = {p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"}
    write_json(output/"artifact_manifest.json", artifacts)
    print(f"Sealed {len(artifacts)} artifacts ({sum(v['bytes'] for v in artifacts.values()):,} bytes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    snapshot(args.output, args.lock_only)
