"""Seal the replay-renewal reports, analysis and reproducibility dependencies."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from acp_cl.replay_renewal.study import write_json


def archive(output):
    root, output = Path(__file__).resolve().parents[1], Path(output).resolve()
    for experiment in ("decomposition", "policy"):
        for name in ("audit.json", "file_integrity.json"):
            if not json.loads((output/experiment/name).read_text(encoding="utf-8"))["passed"]:
                raise ValueError("both complete studies must pass audits before sealing")
    sources = [*sorted((root/"scripts").glob("*replay_renewal*.py")),
        *[root/f"scripts/{name}.py" for name in ("audit_conditional", "audit_training_state",
            "summarize_training_state", "check_training_state_files")],
        *sorted((root/"tests").glob("test_replay_renewal*.py")),
        *sorted((root/"configs").glob("replay_renewal*.json")),
        root/"docs/replay_renewal_protocol.md", root/"docs/ADAM_HANDOFF.md",
        root/"docs/replay_memory_mechanism.md",
        root/"docs/research_charter.md", root/"docs/reproduction.md", root/"reports/replay_renewal_results.md",
        root/"reports/replay_renewal/study_ledger.md", root/"README.md", root/"pyproject.toml",
        root/"requirements-lock.txt"]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with zipfile.ZipFile(output/"analysis_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sources:
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
    args = parser.parse_args()
    archive(args.output)
