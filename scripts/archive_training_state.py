"""Seal training-state analysis, documentation, and completed study artifacts."""

import argparse
import hashlib
from pathlib import Path
import zipfile

from acp_cl.persistence.study import write_json


def archive(output):
    root, output = Path(__file__).resolve().parents[1], Path(output).resolve()
    sources = [*sorted((root/"scripts").glob("*training_state*.py")),
        *[root/f"scripts/{name}.py" for name in ("audit_acquisition", "audit_conditional", "summarize_acquisition")],
        *sorted((root/"tests").glob("test_training_state*.py")),
        *sorted((root/"configs").glob("training_state*.json")),
        root/"docs/training_state_diagnostic.md", root/"docs/sleep_and_training_state.md",
        root/"docs/research_charter.md", root/"docs/reproduction.md", root/"reports/training_state_results.md",
        root/"reports/training_state/development_ledger.md", root/"reports/acquisition_results.md",
        root/"README.md", root/"pyproject.toml", root/".github/workflows/ci.yml"]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with zipfile.ZipFile(output/"analysis_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p, p.relative_to(root).as_posix())
    write_json(output/"analysis_source_manifest.json", hashes)
    artifacts = {p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"}
    write_json(output/"artifact_manifest.json", artifacts)
    print(f"Archived {len(artifacts)} artifacts ({sum(a['bytes'] for a in artifacts.values()):,} bytes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive(args.output)
