"""Archive the contextual analysis and create a verifiable artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

from acp_cl.persistence.study import write_json


def archive(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    files = [*sorted((root/"scripts").glob("*contextual*.py")),
             root/"scripts/audit_conditional.py", root/"scripts/diagnose_conditional_routing.py",
             *sorted((root/"tests").glob("test_contextual*.py")),
             *sorted((root/"configs").glob("contextual*.json")),
             root/"docs/contextual_comparison.md", root/"docs/research_charter.md", root/"docs/reproduction.md",
             root/"reports/contextual_results.md", root/"reports/contextual/development_ledger.md",
             root/"README.md", root/"pyproject.toml", root/".github/workflows/ci.yml"]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output/"analysis_source.zip", "w", zipfile.ZIP_DEFLATED) as handle:
        for source in files:
            handle.write(source, source.relative_to(root).as_posix())
    write_json(output/"analysis_source_manifest.json", hashes)
    artifacts = {p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"}
    write_json(output/"artifact_manifest.json", artifacts)
    print(f"Archived {len(artifacts)} artifacts ({sum(a['bytes'] for a in artifacts.values()):,} bytes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive(args.output)
